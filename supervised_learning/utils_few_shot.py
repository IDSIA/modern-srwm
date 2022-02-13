# Implement evaluation functions

import torch


# eval function for sync-label case
def eval_model_label_sync(model, eval_dataloader, num_steps, device='cuda'):
    val_running_correct = 0
    val_running_total = 0

    for val_batch_id, val_batch in enumerate(eval_dataloader):
        val_inputs, val_targets = val_batch['train']
        val_inputs = val_inputs.to(device=device)  # (B, len, **)
        val_targets = val_targets.to(device=device)  # (B, len)
        val_bsz, _ = val_targets.shape

        val_inputs = val_inputs.transpose(0, 1)
        val_targets = val_targets.transpose(0, 1)

        # 'test' part
        val_test_inputs, val_test_targets = val_batch['test']
        val_test_inputs = val_test_inputs.to(device=device)  # (B, len, **)
        val_test_targets = val_test_targets.to(device=device)  # (B, len)

        val_test_inputs = val_test_inputs.transpose(0, 1)
        val_test_targets = val_test_targets.transpose(0, 1)

        # take just one element
        val_test_inputs = val_test_inputs[0].unsqueeze(0)
        val_test_targets = val_test_targets[0].unsqueeze(0)

        val_net_input = torch.cat([val_inputs, val_test_inputs], dim=0)
        val_target_labels = torch.cat([val_targets, val_test_targets], dim=0)

        with torch.no_grad():
            sync_labels = val_target_labels[:-1]
            dummy_last_token = torch.zeros_like(sync_labels[0].unsqueeze(0))
            label_feedback = torch.cat([sync_labels, dummy_last_token], dim=0)
            outputs, _ = model(val_net_input, label_feedback)
            outputs = outputs[-1]
            _, predicted = outputs.max(-1)
            bool_correct_pred = (predicted == val_target_labels[-1])

            val_running_correct += bool_correct_pred.sum().item()
            val_running_total += val_bsz

        if val_batch_id > num_steps:
            break

    running_correct = val_running_correct / val_running_total

    return running_correct


# eval function for the delayed label case
# compute per-shot average accuracies.
# hard coded for two tasks
def eval_model_delayed_label_multi_sequential(
        model, eval_dataloader0, eval_dataloader1, num_steps, n_way, k_shot,
        device='cuda', state=None):

    running_correct = 0
    running_total = 0

    task_running_correct = {0: 0., 1: 0.}
    counts = 0

    acc_per_shot = {0: [], 1: []}
    cnt_per_shot = {0: [], 1: []}

    for key in acc_per_shot.keys():
        for _ in range(k_shot):
            acc_per_shot[key].append(0)
            cnt_per_shot[key].append(0)

    for batch_id, (batch0, batch1) in enumerate(zip(eval_dataloader0, eval_dataloader1)):
        val_inputs0, val_targets0 = batch0['train']
        val_inputs1, val_targets1 = batch1['train']
        del batch0['test'], batch1['test']

        val_inputs0 = val_inputs0.to(device=device)  # (B, len, **)
        val_targets0 = val_targets0.to(device=device)  # (B, len)
        val_bsz0, val_len0 = val_targets0.shape

        val_inputs1 = val_inputs1.to(device=device)  # (B, len, **)
        val_targets1 = val_targets1.to(device=device)  # (B, len)
        val_bsz1, val_len1 = val_targets1.shape

        val_inputs0 = val_inputs0.transpose(0, 1)
        val_targets0 = val_targets0.transpose(0, 1)

        val_inputs1 = val_inputs1.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        # no trimming needed for eval.
        # contenate along time dimension, alternate order
        if batch_id % 2 == 0:  # ID 0 first
            net_input = torch.cat([val_inputs0, val_inputs1], dim=0)
            target_labels = torch.cat([val_targets0, val_targets1], dim=0)
        else:  # miniimagenet first
            net_input = torch.cat([val_inputs1, val_inputs0], dim=0)
            target_labels = torch.cat([val_targets1, val_targets0], dim=0)

        slen, bsz = target_labels.shape

        delayed_labels = target_labels[:-1]
        dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
        label_feedback = torch.cat([dummy_last_token, delayed_labels], dim=0)

        outputs, _ = model(net_input, label_feedback, state)
        _, predicted = outputs.max(-1)
        bool_correct_pred = (predicted == target_labels)

        running_correct += bool_correct_pred.sum().item()
        running_total += slen * bsz

        if batch_id % 2 == 0:  # ID 0 first
            bool_correct_pred0 = bool_correct_pred[:val_len0]
            bool_correct_pred1 = bool_correct_pred[val_len0:]
        else:
            bool_correct_pred1 = bool_correct_pred[:val_len1]
            bool_correct_pred0 = bool_correct_pred[val_len1:]

        task_running_correct[0] += bool_correct_pred0.sum().item()
        task_running_correct[1] += bool_correct_pred1.sum().item()

        assert val_bsz0 == val_bsz1
        assert val_len0 == val_len1
        counts += val_bsz0 * val_len0  # same size

        val_targets0 = val_targets0.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        bool_correct_pred0 = bool_correct_pred0.transpose(0, 1)
        bool_correct_pred1 = bool_correct_pred1.transpose(0, 1)

        for b in range(bsz):
            # task 0
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets0[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets0[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[0][shot] += (
                            bool_correct_pred0[b][cl_indices[shot]].item())
                        cnt_per_shot[0][shot] += 1
            # task 1
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets1[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets1[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[1][shot] += (
                            bool_correct_pred1[b][cl_indices[shot]].item())
                        cnt_per_shot[1][shot] += 1

        if batch_id > num_steps:
            break

    running_correct = 100 * running_correct / running_total
    task_running_correct[0] = 100 * task_running_correct[0] / counts
    task_running_correct[1] = 100 * task_running_correct[1] / counts

    for key in acc_per_shot.keys():
        for shot in range(k_shot):
            shot_acc = (
                100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
            )
            acc_per_shot[key][shot] = shot_acc

    return running_correct, task_running_correct, acc_per_shot


# eval function for the delayed label case
# compute per-shot & per-position average accuracies
# hard coded for two tasks
def eval_per_pos_model_delayed_label_multi_sequential(
        model, eval_dataloader0, eval_dataloader1, num_steps, n_way, k_shot,
        device='cuda', state=None, omniglot_first=True):

    running_correct = 0
    running_total = 0

    task_running_correct = {0: 0., 1: 0.}
    counts = 0

    acc_per_shot = {0: [], 1: []}  # per positions in this case
    cnt_per_shot = {0: [], 1: []}

    for key in acc_per_shot.keys():
        for _ in range(k_shot):
            acc_per_shot[key].append(0)
            cnt_per_shot[key].append(0)

    acc_per_pos = []  # per positions in this case
    cnt_per_pos = 0

    for _ in range(k_shot * n_way * 2):
        acc_per_pos.append(0)

    for batch_id, (batch0, batch1) in enumerate(zip(eval_dataloader0, eval_dataloader1)):
        val_inputs0, val_targets0 = batch0['train']
        val_inputs1, val_targets1 = batch1['train']
        del batch0['test'], batch1['test']

        val_inputs0 = val_inputs0.to(device=device)  # (B, len, **)
        val_targets0 = val_targets0.to(device=device)  # (B, len)
        val_bsz0, val_len0 = val_targets0.shape

        val_inputs1 = val_inputs1.to(device=device)  # (B, len, **)
        val_targets1 = val_targets1.to(device=device)  # (B, len)
        val_bsz1, val_len1 = val_targets1.shape

        val_inputs0 = val_inputs0.transpose(0, 1)
        val_targets0 = val_targets0.transpose(0, 1)

        val_inputs1 = val_inputs1.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        # no trimming needed for eval.
        # contenate along time dimension, alternate order
        if omniglot_first:  # ID 0 first
            net_input = torch.cat([val_inputs0, val_inputs1], dim=0)
            target_labels = torch.cat([val_targets0, val_targets1], dim=0)
        else:  # miniimagenet first
            net_input = torch.cat([val_inputs1, val_inputs0], dim=0)
            target_labels = torch.cat([val_targets1, val_targets0], dim=0)

        slen, bsz = target_labels.shape

        delayed_labels = target_labels[:-1]
        dummy_last_token = torch.zeros_like(delayed_labels[0].unsqueeze(0))
        label_feedback = torch.cat([dummy_last_token, delayed_labels], dim=0)

        outputs, _ = model(net_input, label_feedback, state)
        _, predicted = outputs.max(-1)
        bool_correct_pred = (predicted == target_labels)

        running_correct += bool_correct_pred.sum().item()
        running_total += slen * bsz

        # per position stats:
        assert slen == k_shot * n_way * 2
        for pos in range(k_shot * n_way * 2):
            acc_per_pos[pos] += bool_correct_pred[pos].sum().item()
        cnt_per_pos += bsz

        if omniglot_first:  # ID 0 first
            bool_correct_pred0 = bool_correct_pred[:val_len0]
            bool_correct_pred1 = bool_correct_pred[val_len0:]
        else:
            bool_correct_pred1 = bool_correct_pred[:val_len1]
            bool_correct_pred0 = bool_correct_pred[val_len1:]

        task_running_correct[0] += bool_correct_pred0.sum().item()
        task_running_correct[1] += bool_correct_pred1.sum().item()

        assert val_bsz0 == val_bsz1
        assert val_len0 == val_len1
        counts += val_bsz0 * val_len0  # same size

        val_targets0 = val_targets0.transpose(0, 1)
        val_targets1 = val_targets1.transpose(0, 1)

        bool_correct_pred0 = bool_correct_pred0.transpose(0, 1)
        bool_correct_pred1 = bool_correct_pred1.transpose(0, 1)

        for b in range(bsz):
            # task 0
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets0[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets0[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[0][shot] += (
                            bool_correct_pred0[b][cl_indices[shot]].item())
                        cnt_per_shot[0][shot] += 1
            # task 1
            prev_cl_end = 0
            _, cnts_uniq = torch.unique(
                val_targets1[b], sorted=True, return_counts=True)
            _, indices = torch.sort(val_targets1[b], stable=True)
            for cl in range(n_way):
                cl_cnts = cnts_uniq[cl]
                cl_indices = indices[prev_cl_end:prev_cl_end + cl_cnts]
                cl_indices_len = len(cl_indices)
                prev_cl_end += cl_cnts

                for shot in range(k_shot):
                    if cl_indices_len > shot:
                        acc_per_shot[1][shot] += (
                            bool_correct_pred1[b][cl_indices[shot]].item())
                        cnt_per_shot[1][shot] += 1

        if batch_id > num_steps:
            break

    running_correct = 100 * running_correct / running_total
    task_running_correct[0] = 100 * task_running_correct[0] / counts
    task_running_correct[1] = 100 * task_running_correct[1] / counts

    for key in acc_per_shot.keys():
        for shot in range(k_shot):
            shot_acc = (
                100 * acc_per_shot[key][shot] / cnt_per_shot[key][shot]
            )
            acc_per_shot[key][shot] = shot_acc

    # per position:
    for pos in range(k_shot * n_way * 2):
        acc_per_pos[pos] = 100 * acc_per_pos[pos] / cnt_per_pos

    return running_correct, task_running_correct, acc_per_shot, acc_per_pos
