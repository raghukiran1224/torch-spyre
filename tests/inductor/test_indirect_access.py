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

    Layout:
        arg 0 (input):   experts tensor  — [num_experts, hidden] — INDIRECT
        arg 1 (input):   indices tensor  — [top_k] — direct, int32
        arg 2 (output):  result tensor   — [top_k, hidden] — direct

    The experts tensor is indirectly addressed: its dim-0 address comes from
    the values in the indices tensor (arg 1).
    """
    k_sym = sympy.Symbol("c0")
    h_sym = sympy.Symbol("c1")
    index_value = sympy.Symbol("index_value")

    experts_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 2, 1],
        device_coordinates=[k_sym, h_sym, sympy.Integer(0)],
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
        device_dtype=DataFormats.IEEE_INT32,
        device_size=[4, 1],
        device_coordinates=[k_sym, sympy.Integer(0)],
        allocation=None,
    )

    output_arg = TensorArg(
        is_input=False,
        arg_index=2,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 2, 1],
        device_coordinates=[k_sym, h_sym, sympy.Integer(0)],
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
        self.assertEqual(isrc.base_offset_expr, "index_value*32768")
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
        self.assertIn("index_value*32768", spec_str)


class TestGenerateSDSCIndirect(unittest.TestCase):
    """Tests that generate_sdsc emits the indirect_src JSON fields."""

    def test_json_contains_indirect_src_on_schedule_tree(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)

        json_str = json.dumps(sdsc_json)
        self.assertIn("indirectSrc_", json_str)

        opfunc_key = list(sdsc_json.keys())[0]
        dsc = sdsc_json[opfunc_key]["dscs_"][0]
        inner_key = list(dsc.keys())[0]
        schedule_tree = dsc[inner_key]["scheduleTree_"]

        found_indirect = False
        for node in schedule_tree:
            if "indirectSrc_" in node:
                found_indirect = True
                isrc = node["indirectSrc_"]
                self.assertEqual(isrc["indexTensorIdx_"], 1)
                self.assertEqual(isrc["baseOffsetExpr_"], "index_value*32768")
                self.assertEqual(isrc["addressMode_"], "ibr")
                break
        self.assertTrue(found_indirect, "No indirectSrc_ found in scheduleTree_")

    def test_json_contains_indirect_access_on_labeled_ds(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)

        opfunc_key = list(sdsc_json.keys())[0]
        dsc = sdsc_json[opfunc_key]["dscs_"][0]
        inner_key = list(dsc.keys())[0]
        labeled_ds = dsc[inner_key]["labeledDs_"]

        found_indirect = False
        for lds in labeled_ds:
            if "indirectAccess_" in lds:
                found_indirect = True
                self.assertEqual(lds["indirectAccess_"]["indexTensorIdx_"], 1)
                self.assertEqual(lds["indirectAccess_"]["addressMode_"], "ibr")
                break
        self.assertTrue(found_indirect, "No indirectAccess_ found in labeledDs_")

    def test_non_indirect_nodes_have_no_indirect_fields(self):
        op_spec = _make_moe_op_spec()
        sdsc_json = compile_op_spec("test_moe_gather", op_spec)

        opfunc_key = list(sdsc_json.keys())[0]
        dsc = sdsc_json[opfunc_key]["dscs_"][0]
        inner_key = list(dsc.keys())[0]
        schedule_tree = dsc[inner_key]["scheduleTree_"]

        for node in schedule_tree:
            if node["ldsIdx_"] != 0:
                self.assertNotIn(
                    "indirectSrc_",
                    node,
                    f"Non-indirect node ldsIdx_={node['ldsIdx_']}"
                    " should not have indirectSrc_",
                )

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
            device_size=[4, 2, 1],
            device_coordinates=[x_sym, y_sym, sympy.Integer(0)],
            allocation=None,
        )
        output_arg = TensorArg(
            is_input=False,
            arg_index=1,
            device_dtype=DataFormats.SEN169_FP16,
            device_size=[4, 2, 1],
            device_coordinates=[x_sym, y_sym, sympy.Integer(0)],
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
        self.assertNotIn("indirectSrc_", json_str)
        self.assertNotIn("indirectAccess_", json_str)


if __name__ == "__main__":
    unittest.main()
