# Copyright 2025 The Torch-Spyre Authors.
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

"""Tests for indirect access (data-dependent addressing) in OpSpec and SDSC.

These tests validate the MoE indirect access pipeline from OpSpec through
SDSC JSON emission.  They run without Spyre hardware — only the Python
package and its C extension are needed (available in the pod venv).
"""

import json
import unittest

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import (
    IndirectSource,
    OpSpec,
    TensorArg,
)
from torch_spyre._inductor.codegen.superdsc import (
    SDSCIndirectSrc,
    parse_op_spec,
    compile_op_spec,
)


def _make_moe_op_spec():
    """Construct an OpSpec modelling a simplified MoE expert gather.

    Layout (fp16, elems_per_stick=64):
        arg 0 (input):   experts — logically [num_experts=8, hidden=128]
                          device_size = [hidden_sticks=2, num_experts=8, 64]
                          INDIRECT on dim 0 (expert selection via indices)
        arg 1 (input):   indices — logically [top_k=4]
                          device_size = [1, top_k=4, 64]
        arg 2 (output):  result  — logically [top_k=4, hidden=128]
                          device_size = [hidden_sticks=2, top_k=4, 64]

    device_coordinates follow the convention:
        [floor(h/64), k, Mod(h, 64)] — last coord is within-stick position.
    """
    k_sym = sympy.Symbol("c0")
    h_sym = sympy.Symbol("c1")
    index_value = sympy.Symbol("index_value")

    experts_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 8, 64],
        device_coordinates=[
            sympy.floor(h_sym / 64),
            k_sym,
            sympy.Mod(h_sym, 64),
        ],
        allocation=None,
        indirect_source=IndirectSource(
            index_arg_index=1,
            gather_dim=0,
            base_offset_expr=index_value * 32768,
        ),
    )

    indices_arg = TensorArg(
        is_input=True,
        arg_index=1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[1, 4, 64],
        device_coordinates=[
            sympy.Integer(0),
            k_sym,
            sympy.Integer(0),
        ],
        allocation=None,
    )

    output_arg = TensorArg(
        is_input=False,
        arg_index=2,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 4, 64],
        device_coordinates=[
            sympy.floor(h_sym / 64),
            k_sym,
            sympy.Mod(h_sym, 64),
        ],
        allocation=None,
    )

    iteration_space = {
        k_sym: (sympy.Integer(4), 1),
        h_sym: (sympy.Integer(128), 1),
    }

    return OpSpec(
        op="identity",
        is_reduction=False,
        iteration_space=iteration_space,
        args=[experts_arg, indices_arg, output_arg],
        op_info={},
    )


class TestIndirectSourceDataclass(unittest.TestCase):
    """Basic tests for the IndirectSource dataclass."""

    def test_create_indirect_source(self):
        iv = sympy.Symbol("index_value")
        src = IndirectSource(
            index_arg_index=1,
            gather_dim=0,
            base_offset_expr=iv * 256,
        )
        self.assertEqual(src.index_arg_index, 1)
        self.assertEqual(src.gather_dim, 0)
        self.assertEqual(src.base_offset_expr, iv * 256)

    def test_tensor_arg_default_no_indirect(self):
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 2, 1],
            device_coordinates=[sympy.Symbol("x"), sympy.Integer(0)],
            allocation=None,
        )
        self.assertIsNone(arg.indirect_source)

    def test_tensor_arg_with_indirect(self):
        iv = sympy.Symbol("index_value")
        src = IndirectSource(index_arg_index=2, gather_dim=0, base_offset_expr=iv * 64)
        arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[8, 2, 1],
            device_coordinates=[sympy.Symbol("x"), sympy.Integer(0)],
            allocation=None,
            indirect_source=src,
        )
        self.assertIsNotNone(arg.indirect_source)
        self.assertEqual(arg.indirect_source.index_arg_index, 2)


class TestSDSCIndirectSrc(unittest.TestCase):
    """Tests for SDSCIndirectSrc dataclass."""

    def test_defaults(self):
        s = SDSCIndirectSrc(index_tensor_idx=1, base_offset_expr="index_value*32768")
        self.assertEqual(s.address_mode, "ibr")

    def test_custom_address_mode(self):
        s = SDSCIndirectSrc(
            index_tensor_idx=1,
            base_offset_expr="index_value*32768",
            address_mode="direct",
        )
        self.assertEqual(s.address_mode, "direct")


class TestParseOpSpecIndirect(unittest.TestCase):
    """Tests that parse_op_spec correctly propagates indirect source info."""

    def test_indirect_source_propagated_to_sdsc_args(self):
        op_spec = _make_moe_op_spec()
        sdsc_spec = parse_op_spec(op_spec)

        indirect_args = [a for a in sdsc_spec.args if a.indirect_src is not None]
        self.assertEqual(len(indirect_args), 1, "Exactly one arg should be indirect")

        isrc = indirect_args[0].indirect_src
        self.assertEqual(isrc.index_tensor_idx, 1)
        self.assertIn("index_value", isrc.base_offset_expr)
        self.assertIn("32768", isrc.base_offset_expr)
        self.assertEqual(isrc.address_mode, "ibr")

    def test_non_indirect_args_have_no_indirect_src(self):
        op_spec = _make_moe_op_spec()
        sdsc_spec = parse_op_spec(op_spec)

        for i, arg in enumerate(sdsc_spec.args):
            if i == 0:
                continue
            self.assertIsNone(
                arg.indirect_src,
                f"Arg {i} should not have indirect_src",
            )

    def test_sdsc_spec_str_includes_indirect(self):
        op_spec = _make_moe_op_spec()
        sdsc_spec = parse_op_spec(op_spec)
        spec_str = str(sdsc_spec)
        self.assertIn("indirect_tensors", spec_str)
        self.assertIn("index_value", spec_str)
        self.assertIn("32768", spec_str)


class TestGenerateSDSCIndirect(unittest.TestCase):
    """Tests that generate_sdsc emits deeptools-compatible indirect access fields."""

    def _get_schedule_tree(self, sdsc_json):
        opfunc_key = list(sdsc_json.keys())[0]
        dsc = sdsc_json[opfunc_key]["dscs_"][0]
        inner_key = list(dsc.keys())[0]
        return dsc[inner_key]["scheduleTree_"]

    def test_value_tensor_has_indirect_alloc_type(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)
        schedule_tree = self._get_schedule_tree(sdsc_json)

        experts_node = schedule_tree[0]
        self.assertEqual(experts_node["ldsIdx_"], 0)
        self.assertEqual(experts_node["indirectAllocType_"], "value_tensor")
        self.assertIn("relatedIndirectAccessAlloc_", experts_node)

    def test_index_tensor_has_indirect_alloc_type(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)
        schedule_tree = self._get_schedule_tree(sdsc_json)

        indices_node = schedule_tree[1]
        self.assertEqual(indices_node["ldsIdx_"], 1)
        self.assertEqual(indices_node["indirectAllocType_"], "index_tensor")
        self.assertIn("relatedIndirectAccessAlloc_", indices_node)

    def test_related_alloc_cross_references(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)
        schedule_tree = self._get_schedule_tree(sdsc_json)

        experts_node = schedule_tree[0]
        indices_node = schedule_tree[1]
        self.assertEqual(
            experts_node["relatedIndirectAccessAlloc_"],
            indices_node["name_"],
        )
        self.assertEqual(
            indices_node["relatedIndirectAccessAlloc_"],
            experts_node["name_"],
        )

    def test_output_tensor_has_no_indirection(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)
        schedule_tree = self._get_schedule_tree(sdsc_json)

        output_node = schedule_tree[2]
        self.assertEqual(output_node["indirectAllocType_"], "no_indirection")
        self.assertNotIn("relatedIndirectAccessAlloc_", output_node)

    def test_json_is_valid_and_serializable(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)
        serialized = json.dumps(sdsc_json, indent=2)
        roundtrip = json.loads(serialized)
        self.assertEqual(sdsc_json, roundtrip)

    def test_standard_op_spec_no_indirect_fields(self):
        """An OpSpec with no indirect sources produces JSON with no indirect fields."""
        x_sym = sympy.Symbol("c0")
        y_sym = sympy.Symbol("c1")

        input_arg = TensorArg(
            is_input=True,
            arg_index=0,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 4, 64],
            device_coordinates=[
                sympy.floor(y_sym / 64),
                x_sym,
                sympy.Mod(y_sym, 64),
            ],
            allocation=None,
        )
        output_arg = TensorArg(
            is_input=False,
            arg_index=1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[2, 4, 64],
            device_coordinates=[
                sympy.floor(y_sym / 64),
                x_sym,
                sympy.Mod(y_sym, 64),
            ],
            allocation=None,
        )
        op_spec = OpSpec(
            op="identity",
            is_reduction=False,
            iteration_space={
                x_sym: (sympy.Integer(4), 1),
                y_sym: (sympy.Integer(128), 1),
            },
            args=[input_arg, output_arg],
            op_info={},
        )

        sdsc_json = compile_op_spec("test_standard", op_spec)
        json_str = json.dumps(sdsc_json)
        self.assertNotIn("value_tensor", json_str)
        self.assertNotIn("index_tensor", json_str)
        self.assertNotIn("relatedIndirectAccessAlloc_", json_str)


if __name__ == "__main__":
    unittest.main()
