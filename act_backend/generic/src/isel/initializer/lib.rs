use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

use egg::{EGraph, Id};

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};
use crate::ir::metadata::{MetaData, MetaDataInfo};
use crate::isel::initializer::parser::HLOExpr;

use crate::isel::initializer::parser::{
  parse_hlo_module_with_regions, parse_type_shape, HloModuleHeader,
};

use crate::isel::initializer::nodes::{
  add_binary_op, add_broadcast, add_concatenate, add_constant, add_convert, add_exponential,
  add_parameter, add_reduce, add_reshape, add_root_operations, add_slice, add_transpose,
};
use crate::isel::initializer::shapes::shape_to_vec;

pub type EggSymbolTable = HashMap<String, Id>;

// A kernel's e-graph, HBM offsets, root, inputs and metadata
pub type ParsedModule = (
  EGraph<TensorOp, TensorInfo>,
  Vec<(Option<Id>, i32)>,
  Id,
  HashSet<Id>,
  MetaData,
);

pub fn parse_hlo_module_to_egraph<P: AsRef<Path>>(path: P) -> Result<ParsedModule, io::Error> {
  let mut egraph = EGraph::<TensorOp, TensorInfo>::default();
  let mut inputs = HashSet::default();
  let mut table = EggSymbolTable::default();
  let mut offsets: Vec<(Option<Id>, i32)> = vec![];
  let mut metadata: MetaData = MetaData::default();

  let file = File::open(path)?;
  let reader = BufReader::new(file);

  // Read all lines into a vector
  let lines: Vec<String> = reader.lines().collect::<Result<Vec<_>, _>>()?;
  let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();

  // Parse the module with regions
  if let Some((header, entry_instructions)) = parse_hlo_module_with_regions(&line_refs) {
    // Fill offsets from the header
    fill_offsets_from_header(&header, &mut offsets, &mut metadata);

    // Process entry instructions
    let mut root: Option<Id> = None;
    let total_instructions = entry_instructions.len();

    for (i, expr) in entry_instructions.into_iter().enumerate() {
      // The last instruction is typically the ROOT operation
      let is_root = i == total_instructions - 1;

      add_expr_to_egraph(
        &expr,
        &mut egraph,
        &mut inputs,
        &mut table,
        &mut offsets,
        &mut root,
        is_root,
      );
    }

    if let Some(last) = offsets.last_mut() {
      last.0 = root;
    }

    Ok((egraph, offsets, root.unwrap(), inputs, metadata))
  } else {
    panic!("Failed to parse HLO module with regions");
  }
}

fn fill_offsets_from_header(
  header: &HloModuleHeader,
  offsets: &mut Vec<(Option<Id>, i32)>,
  metadata: &mut MetaData,
) {
  metadata.module_name = header.module_name.clone();

  let mut curr_bytes: i32 = 0;
  for shape in header.input_types.iter() {
    let (ttype, tshape, _) = parse_type_shape(shape);
    let dtype: Dtype = ttype.as_str().into();
    let shape_vec = shape_to_vec(&tshape);

    let mut bytes: i32 = dtype.size_in_bytes();
    shape_vec.iter().for_each(|dim| bytes *= dim);

    offsets.push((None, curr_bytes));
    metadata.input.push(MetaDataInfo {
      addr: curr_bytes,
      shape: shape_vec,
      dtype,
    });

    curr_bytes += bytes;
  }

  let output_shape = &header.output_type;
  let (ttype, tshape, _) = parse_type_shape(output_shape);
  let dtype: Dtype = ttype.as_str().into();
  let shape_vec = shape_to_vec(&tshape);

  offsets.push((None, curr_bytes));
  metadata.output.push(MetaDataInfo {
    addr: curr_bytes,
    shape: shape_vec.clone(),
    dtype,
  });
}

fn add_expr_to_egraph(
  expr: &HLOExpr,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  inputs: &mut HashSet<Id>,
  table: &mut EggSymbolTable,
  offsets: &mut [(Option<Id>, i32)],
  root: &mut Option<Id>,
  parsing_root: bool,
) {
  let (tsymbol, ttype, tshape, ttile) = match expr {
    HLOExpr::Parameter {
      tsymbol,
      tshape,
      ttile,
      ttype,
      tidx,
    } => {
      assert!(!parsing_root); // parameters should not be in root
      if offsets[*tidx as usize].0.is_none() {
        offsets[*tidx as usize].0 = Some(add_parameter(
          tsymbol, tshape, ttile, ttype, egraph, inputs, table,
        ));
      } else {
        panic!("Duplicate parameter index: {}", tidx);
      }
      return;
    }

    HLOExpr::Constant {
      tsymbol,
      ttype,
      tshape,
      value,
    } => {
      assert!(!parsing_root); // constants should not be in root
      add_constant(tsymbol, ttype, tshape, value, egraph, table);
      return;
    }

    HLOExpr::Broadcast {
      tsymbol,
      ttype,
      tshape,
      ttile,
      source,
    } => {
      add_broadcast(tsymbol, ttype, tshape, source, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::BinaryOp {
      tsymbol,
      ttype,
      tshape,
      op_name,
      ttile,
      lhs,
      rhs,
    } => {
      add_binary_op(tsymbol, ttype, tshape, op_name, lhs, rhs, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::Slice {
      tsymbol,
      ttype,
      tshape,
      source,
      ttile,
      slices,
    } => {
      add_slice(tsymbol, ttype, tshape, source, slices, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::Concatenate {
      tsymbol,
      ttype,
      tshape,
      ttile,
      tinputs,
      dimensions,
    } => {
      add_concatenate(tsymbol, ttype, tshape, tinputs, dimensions, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::Reshape {
      tsymbol,
      ttype,
      tshape,
      ttile,
      source,
    } => {
      add_reshape(tsymbol, ttype, tshape, source, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::Transpose {
      tsymbol,
      ttype,
      tshape,
      ttile,
      source,
      dimensions,
    } => {
      add_transpose(tsymbol, ttype, tshape, source, dimensions, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::Convert {
      tsymbol,
      ttype,
      tshape,
      ttile,
      source,
    } => {
      add_convert(tsymbol, ttype, tshape, source, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::ReduceSum {
      tsymbol,
      ttype,
      tshape,
      ttile,
      source,
      dimensions,
    } => {
      add_reduce(tsymbol, ttype, tshape, source, dimensions, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }

    HLOExpr::Exponential {
      tsymbol,
      ttype,
      tshape,
      ttile,
      source,
    } => {
      add_exponential(tsymbol, ttype, tshape, source, egraph, table);
      (tsymbol, ttype, tshape, ttile)
    }
  };

  // The last instruction is written back as bytes
  if parsing_root {
    add_root_operations(tsymbol, ttype, tshape, ttile, egraph, table, root);
  }
}
