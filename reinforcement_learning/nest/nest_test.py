# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import unittest

import nest
import torch


class NestTest(unittest.TestCase):
    def setUp(self):
        self.n1 = ("Test", ["More", 32], {"h": 4})
        self.n2 = ("Test", ("More", 32, (None, 43, ())), {"h": 4})

    def test_nest_flatten_no_asserts(self):
        t = torch.tensor(1)
        t2 = torch.tensor(2)
        n = (t, t2)
        d = {"hey": t}

        nest.flatten((t, t2))
        nest.flatten(d)
        nest.flatten((d, t))
        nest.flatten((d, n, t))

        nest.flatten(((t, t2), (t, t2)))

        nest.flatten(self.n1)
        nest.flatten(self.n2)

        d2 = {"hey": t2, "there": d, "more": t2}
        nest.flatten(d2)  # Careful here, order not necessarily as above.

    def test_nest_map(self):
        t1 = torch.tensor(0)
        t2 = torch.tensor(1)
        d = {"hey": t2}

        n = nest.map(lambda t: t + 42, (t1, t2))

        self.assertSequenceEqual(n, [t1 + 42, t2 + 42])
        self.assertSequenceEqual(n, nest.flatten(n))

        n1 = (d, n, t1)
        n2 = nest.map(lambda t: t * 2, n1)

        self.assertEqual(n2[0], {"hey": torch.tensor(2)})
        self.assertEqual(n2[1], (torch.tensor(84), torch.tensor(86)))
        self.assertEqual(n2[2], torch.tensor(0))

        t = torch.tensor(42)

        # Doesn't work with pybind11/functional.h, but does with py::function.
        self.assertEqual(nest.map(t.add, t2), torch.tensor(43))

    def test_nest_flatten(self):
        self.assertEqual(nest.flatten(None), [None])
        self.assertEqual(nest.flatten(self.n1), ["Test", "More", 32, 4])

    def test_nest_pack_as(self):
        self.assertEqual(self.n2, nest.pack_as(self.n2, nest.flatten(self.n2)))

        with self.assertRaisesRegex(ValueError, "didn't exhaust sequence"):
            nest.pack_as(self.n2, nest.flatten(self.n2) + [None])
        with self.assertRaisesRegex(ValueError, "Too few elements"):
            nest.pack_as(self.n2, nest.flatten(self.n2)[1:])

    def test_nest_map_many2(self):
        def f(a, b):
            return (b, a)

        self.assertEqual(nest.map_many2(f, (1, 2), (3, 4)), ((3, 1), (4, 2)))

        with self.assertRaisesRegex(ValueError, "got 2 vs 1"):
            nest.map_many2(f, (1, 2), (3,))

        self.assertEqual(nest.map_many2(f, {"a": 1}, {"a": 2}), {"a": (2, 1)})

        with self.assertRaisesRegex(ValueError, "same keys"):
            nest.map_many2(f, {"a": 1}, {"b": 2})

        with self.assertRaisesRegex(ValueError, "1 vs 0"):
            nest.map_many2(f, {"a": 1}, {})

        with self.assertRaisesRegex(ValueError, "nests don't match"):
            nest.map_many2(f, {"a": 1}, ())

    def test_nest_map_many(self):
        def f(a):
            return (a[1], a[0])

        self.assertEqual(nest.map_many(f, (1, 2), (3, 4)), ((3, 1), (4, 2)))

        return
        with self.assertRaisesRegex(ValueError, "got 2 vs 1"):
            nest.map_many(f, (1, 2), (3,))

        self.assertEqual(nest.map_many(f, {"a": 1}, {"a": 2}), {"a": (2, 1)})

        with self.assertRaisesRegex(ValueError, "same keys"):
            nest.map_many(f, {"a": 1}, {"b": 2})

        with self.assertRaisesRegex(ValueError, "1 vs 0"):
            nest.map_many(f, {"a": 1}, {})

        with self.assertRaisesRegex(ValueError, "nests don't match"):
            nest.map_many(f, {"a": 1}, ())

    def test_front(self):
        self.assertEqual(nest.front((1, 2, 3)), 1)
        self.assertEqual(nest.front((2, 3)), 2)
        self.assertEqual(nest.front((3,)), 3)

    def test_refcount(self):
        obj = "my very large and random string with numbers 1234"

        rc = sys.getrefcount(obj)

        # Test nest.front. This doesn't involve returning nests
        # from C++ to Python.
        nest.front((None, obj))
        self.assertEqual(rc, sys.getrefcount(obj))

        nest.front(obj)
        self.assertEqual(rc, sys.getrefcount(obj))

        nest.front((obj,))
        self.assertEqual(rc, sys.getrefcount(obj))

        nest.front((obj, obj, [obj, {"obj": obj}, obj]))
        self.assertEqual(rc, sys.getrefcount(obj))

        # Test returning nests of Nones.
        nest.map(lambda x: None, (obj, obj, [obj, {"obj": obj}, obj]))
        self.assertEqual(rc, sys.getrefcount(obj))

        # Test returning actual nests.
        nest.map(lambda s: s, obj)
        self.assertEqual(rc, sys.getrefcount(obj))

        nest.map(lambda x: x, {"obj": obj})
        self.assertEqual(rc, sys.getrefcount(obj))

        nest.map(lambda x: x, (obj,))
        self.assertEqual(rc, sys.getrefcount(obj))

        nest.map(lambda s: s, (obj, obj))
        nest.map(lambda s: s, (obj, obj))
        self.assertEqual(rc, sys.getrefcount(obj))

        n = nest.map(lambda s: s, (obj,))
        self.assertEqual(rc + 1, sys.getrefcount(obj))
        del n
        self.assertEqual(rc, sys.getrefcount(obj))


if __name__ == "__main__":
    unittest.main()
