//! The e-graph nodes of each kernel line, by its operation

use std::collections::HashSet;

use egg::{EGraph, Id};

use crate::ir::dtype::Dtype;
use crate::ir::egraph::{TensorInfo, TensorOp};

use crate::isel::initializer::lib::EggSymbolTable;
use crate::isel::initializer::shapes::{
  apply_transpose_order, get_bytes, get_result_shape, get_tiled_shape, get_transpose_order_param,
  get_transpose_order_result, shape_to_vec,
};

macro_rules! binary_op_match {
    ($op_name:expr, $lhs:expr, $rhs:expr, $table:expr, $egraph:expr, {
        $($key:literal => $variant:path),* $(,)?
    }) => {
        match $op_name {
            $(
                $key => {
                    $egraph.add($variant([
                        lookup($table, $lhs)?,
                        lookup($table, $rhs)?,
                    ]))
                }
            )*
            // The parser builds a binary operation for these words alone
            other => unreachable!("binary operation {}", other),
        }
    };
}

// A kernel name's e-class, or why none is defined
fn lookup(table: &EggSymbolTable, symbol: &str) -> Result<Id, String> {
  table.get(symbol).copied().ok_or_else(|| {
    format!(
      "the kernel reads {}, which no earlier line of it defines",
      symbol
    )
  })
}

// Give a node its type, shape and kernel name
pub(super) fn bind_node(
  tsymbol: &str,
  node: Id,
  shape: Vec<i32>,
  dtype: Dtype,
  is_const: bool,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) {
  egraph.set_analysis_data(
    node,
    TensorInfo {
      shape,
      dtype,
      is_const,
    },
  );
  table.insert(tsymbol.to_string(), node);
}

pub fn add_parameter(
  tsymbol: &str,
  tshape: &str,
  ttile: &str,
  ttype: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  inputs: &mut HashSet<Id>,
  table: &mut EggSymbolTable,
) -> Result<Id, String> {
  let dtype = Dtype::try_from(ttype)?;
  let dtype_size = dtype.size_in_bytes();

  let bytes = get_bytes(tshape, dtype_size)?;
  if ttile.is_empty() {
    // no tile supplied case
    let var_node = egraph.add(TensorOp::Var(tsymbol.to_string()));
    egraph.set_analysis_data(
      var_node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let mut shape_with_size = tshape.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [var_node]));
    let reshape_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(tshape)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    table.insert(tsymbol.to_string(), node);
    inputs.insert(var_node);
    Ok(var_node)
  } else {
    let var_node = egraph.add(TensorOp::Var(tsymbol.to_string()));
    egraph.set_analysis_data(
      var_node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let tiled_shape = get_tiled_shape(tshape, ttile)?;

    let mut shape_with_size = tiled_shape.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [var_node]));
    let reshape_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(&tiled_shape)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    let transpose_order = get_transpose_order_param(tshape.split(',').count() as i32);
    let node = egraph.add(TensorOp::OpTranspose(transpose_order.clone(), [node]));
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    let mut shape_with_size = tshape.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = egraph.add(TensorOp::OpReshape(shape_with_size.clone(), [node]));
    let reshape_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    table.insert(tsymbol.to_string(), node);
    inputs.insert(var_node);
    Ok(var_node)
  }
}

pub(super) fn add_constant(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  value: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpConstant(value.to_string()));
  bind_node(tsymbol, node, shape, dtype, true, egraph, table);
  Ok(())
}

pub(super) fn add_broadcast(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpBroadcast(
    tshape.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

// One argument per field of the kernel's line
#[allow(clippy::too_many_arguments)]
pub(super) fn add_binary_op(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  op_name: &str,
  lhs: &str,
  rhs: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = binary_op_match!(op_name, lhs, rhs, table, egraph, {
      "add" => TensorOp::OpAdd,
      "xor" => TensorOp::OpXor,
      "shift-right-logical" => TensorOp::OpShiftRightLogical,
      "or" => TensorOp::OpOr,
      "shift-left" => TensorOp::OpShiftLeft,
      "dot" => TensorOp::OpDot,
      "divide" => TensorOp::OpDivide
  });
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_slice(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  slices: &[String],
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let slice_str = slices
    .iter()
    .map(|s| {
      let trimmed = s.trim_matches(|c| c == '[' || c == ']');
      // remove stride component - only keep start:end
      let parts: Vec<&str> = trimmed.split(':').collect();
      if parts.len() >= 2 {
        format!("{}:{}", parts[0], parts[1])
      } else {
        trimmed.to_string()
      }
    })
    .collect::<Vec<String>>()
    .join(",");

  let node = egraph.add(TensorOp::OpSlice(
    slice_str.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_concatenate(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  inputs: &[String],
  dimensions: &[u32],
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  if inputs.len() != 2 {
    let values = if inputs.len() == 1 { "value" } else { "values" };
    return Err(format!(
      "{} concatenates {} {}, where a concatenate takes 2",
      tsymbol,
      inputs.len(),
      values
    ));
  }
  let dim_str = dimensions
    .iter()
    .map(|d| d.to_string())
    .collect::<Vec<String>>()
    .join(",");
  let node = egraph.add(TensorOp::OpConcat(
    dim_str.clone(),
    [lookup(table, &inputs[0])?, lookup(table, &inputs[1])?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_reshape(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpReshape(
    tshape.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_transpose(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  dimensions: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpTranspose(
    dimensions.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_convert(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpConvert(
    ttype.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_reduce(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  dimensions: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpReduceSum(
    dimensions.to_string(),
    [lookup(table, source)?],
  ));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_exponential(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  source: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let shape = shape_to_vec(tshape)?;

  let node = egraph.add(TensorOp::OpExp([lookup(table, source)?]));
  bind_node(tsymbol, node, shape, dtype, false, egraph, table);
  Ok(())
}

pub(super) fn add_root_operations(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  ttile: &str,
  egraph: &mut EGraph<TensorOp, TensorInfo>,
  table: &mut EggSymbolTable,
  root: &mut Option<Id>,
) -> Result<(), String> {
  let dtype = Dtype::try_from(ttype)?;
  let dtype_size = dtype.size_in_bytes();
  let bytes = get_bytes(tshape, dtype_size)?;

  if ttile.is_empty() {
    let mut shape_with_size = tshape.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = lookup(table, tsymbol)?;

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpReshape(bytes.to_string(), [node]));
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    table.insert(tsymbol.to_string(), node);
    *root = Some(node);
  } else {
    let node = lookup(table, tsymbol)?;
    let tiled_shape = get_result_shape(tshape, ttile)?;

    let node = egraph.add(TensorOp::OpReshape(tiled_shape.clone(), [node]));
    let reshape_vec = shape_to_vec(&tiled_shape)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: reshape_vec.clone(),
        dtype,
        is_const: false,
      },
    );

    let transpose_order = get_transpose_order_result(tshape.split(',').count() as i32);
    let node = egraph.add(TensorOp::OpTranspose(transpose_order.clone(), [node]));

    let transposed_reshape_str = apply_transpose_order(&tiled_shape, &transpose_order);
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: shape_to_vec(&transposed_reshape_str)?,
        dtype,
        is_const: false,
      },
    );

    let mut shape_with_size = transposed_reshape_str.to_string();
    if dtype_size != 1 {
      shape_with_size += &format!(",{}", dtype_size);
    }

    let node = egraph.add(TensorOp::OpBitcvt([node]));
    let bitcvt_vec = shape_to_vec(&shape_with_size)?;
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: bitcvt_vec.clone(),
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    let node = egraph.add(TensorOp::OpReshape(bytes.to_string(), [node]));
    egraph.set_analysis_data(
      node,
      TensorInfo {
        shape: vec![bytes],
        dtype: Dtype::U8,
        is_const: false,
      },
    );

    table.insert(tsymbol.to_string(), node);
    *root = Some(node);
  }
  Ok(())
}
