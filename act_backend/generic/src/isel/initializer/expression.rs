use crate::isel::initializer::parser::HLORegion;
use crate::isel::initializer::shapes::shape_text;

#[derive(Debug, Clone)]
pub enum HLOExpr {
  Parameter {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    tidx: u32,
  },
  Constant {
    tsymbol: String,
    ttype: String,
    tshape: String,
    value: String,
  },
  Broadcast {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    source: String,
    dimensions: String,
  },
  Slice {
    tsymbol: String,
    ttype: String,
    tshape: String,
    source: String,
    ttile: String,
    slices: Vec<String>,
  },
  BinaryOp {
    tsymbol: String,
    ttype: String,
    tshape: String,
    op_name: String,
    ttile: String,
    lhs: String,
    rhs: String,
  },
  Dot {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    lhs: String,
    rhs: String,
    dimensions: String,
  },
  Concatenate {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    tinputs: Vec<String>,
    dimensions: Vec<u32>,
  },
  Reshape {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    source: String,
  },
  Transpose {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    source: String,
    dimensions: String,
  },
  Convert {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    source: String,
  },
  ReduceSum {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    source: String,
    init: String,
    dimensions: String,
  },
  Exponential {
    tsymbol: String,
    ttype: String,
    tshape: String,
    ttile: String,
    source: String,
  },
}

pub fn validate_region_exists(
  region_name: &str,
  regions: &std::collections::HashMap<String, HLORegion>,
) -> Result<(), String> {
  let Some(region) = regions.get(region_name) else {
    return Err(format!(
      "a reduce applies the region {}, which the kernel does not define",
      region_name
    ));
  };

  // region only contains parameter and add instructions
  for instruction in region.instructions.iter() {
    match instruction {
      HLOExpr::Parameter { .. } => {}
      HLOExpr::BinaryOp { op_name, .. } if op_name == "add" => {}
      _ => {
        return Err(format!(
          "the region {} computes something other than a sum, the only reduce the backend supports",
          region_name
        ));
      }
    }
  }
  Ok(())
}

// A line's attributes after its operands, as names and values
fn read_attributes(trest: &str) -> Vec<(String, String)> {
  let mut attributes = vec![];
  let mut piece = String::new();
  let mut depth = 0;
  let mut quoted = false;
  // Commas inside brackets or quotes belong to a value
  for c in trest.trim().trim_start_matches(',').chars().chain([',']) {
    match c {
      '"' => quoted = !quoted,
      '{' | '[' | '(' if !quoted => depth += 1,
      '}' | ']' | ')' if !quoted => depth -= 1,
      ',' if depth == 0 && !quoted => {
        let text = std::mem::take(&mut piece);
        if text.trim().is_empty() {
          continue;
        }
        // A piece with no name is skipped
        if let Some((name, value)) = text.split_once('=') {
          attributes.push((name.trim().to_string(), value.trim().to_string()));
        }
        continue;
      }
      _ => {}
    }
    piece.push(c);
  }
  attributes
}

// The value of a required attribute, its braces taken off
fn required<'a>(
  tsymbol: &str,
  toperator: &str,
  attributes: &'a [(String, String)],
  name: &str,
) -> Result<&'a str, String> {
  attributes
    .iter()
    .find(|(given, _)| given == name)
    .map(|(_, value)| value.trim_start_matches('{').trim_end_matches('}').trim())
    .ok_or_else(|| format!("{} gives {} no {} attribute", tsymbol, toperator, name))
}

// A dot's dimension lists, in its payload's order, and whether each is required
const DOT_DIMENSIONS: [(&str, bool); 4] = [
  ("lhs_batch_dims", false),
  ("lhs_contracting_dims", true),
  ("rhs_batch_dims", false),
  ("rhs_contracting_dims", true),
];

// A list of dimensions, or why an entry is not one
fn dimension_list(tsymbol: &str, name: &str, text: &str) -> Result<Vec<i32>, String> {
  let mut dimensions = vec![];
  for text in text
    .split(',')
    .map(str::trim)
    .filter(|text| !text.is_empty())
  {
    match text.parse::<i32>() {
      Ok(dimension) if dimension >= 0 => dimensions.push(dimension),
      _ => {
        return Err(format!(
          "{} gives '{}' in {}, which is not a dimension",
          tsymbol, text, name
        ))
      }
    }
  }
  Ok(dimensions)
}

// One argument per field of the kernel's line
#[allow(clippy::too_many_arguments)]
pub fn parse_hlo_expr(
  tsymbol: &str,
  ttype: &str,
  tshape: &str,
  ttile: &str,
  toperator: &str,
  toperands: &str,
  trest: &str,
  regions: &std::collections::HashMap<String, HLORegion>,
  shapes: &std::collections::HashMap<String, String>,
) -> Result<HLOExpr, String> {
  let attributes = read_attributes(trest);

  let expr = match toperator {
    "parameter" => {
      // Parameter(0) -- parse the index from operands
      let Ok(tidx) = toperands.trim().parse::<u32>() else {
        return Err(format!(
          "{} reads parameter {}, which is not a parameter number",
          tsymbol,
          toperands.trim()
        ));
      };
      HLOExpr::Parameter {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        tidx,
      }
    }

    "constant" => {
      // constant(1002)
      let value = toperands.trim().to_string();
      HLOExpr::Constant {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        value,
      }
    }

    "broadcast" => {
      // broadcast(Arg_0.6), dimensions={0}
      let source = toperands.trim().to_string();

      // The result dimension each source dimension lands on
      let dim_str = match required(tsymbol, toperator, &attributes, "dimensions") {
        // XLA-HLO reads a rank-0 source's missing map as {}
        Err(_) if shapes.get(&source).is_some_and(|shape| shape == "[]") => "",
        given => given?,
      };
      let dimensions = dimension_list(tsymbol, "dimensions", dim_str)?;

      HLOExpr::Broadcast {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        dimensions: shape_text(&dimensions),
      }
    }

    "slice" => {
      // slice(source), slice={[0:1:1], [0:8:1]}
      let source = toperands.trim().to_string();

      // One [start:limit:stride] per dimension
      let slices = required(tsymbol, toperator, &attributes, "slice")?
        .split(',')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();

      HLOExpr::Slice {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        slices,
      }
    }

    "add" | "xor" | "shift-right-logical" | "or" | "shift-left" | "dot" | "divide" => {
      // binary operators with two operands separated by comma
      let operands: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
      if operands.len() != 2 {
        let values = if operands.len() == 1 {
          "value"
        } else {
          "values"
        };
        return Err(format!(
          "{} applies {} to {} {}, where {} takes 2",
          tsymbol,
          toperator,
          operands.len(),
          values,
          toperator
        ));
      }
      let lhs = operands[0].to_string();
      let rhs = operands[1].to_string();

      if toperator == "dot" {
        // dot(%a, %b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
        let mut lists = vec![];
        for (name, needed) in DOT_DIMENSIONS {
          // A batch list left out is empty
          if !needed && !attributes.iter().any(|(given, _)| given == name) {
            lists.push(vec![]);
            continue;
          }
          let text = required(tsymbol, toperator, &attributes, name)?;
          lists.push(dimension_list(tsymbol, name, text)?);
        }
        HLOExpr::Dot {
          tsymbol: tsymbol.to_string(),
          ttype: ttype.to_string(),
          tshape: tshape.to_string(),
          ttile: ttile.to_string(),
          lhs,
          rhs,
          dimensions: lists
            .iter()
            .map(|list| shape_text(list))
            .collect::<Vec<_>>()
            .join(";"),
        }
      } else {
        HLOExpr::BinaryOp {
          tsymbol: tsymbol.to_string(),
          ttype: ttype.to_string(),
          tshape: tshape.to_string(),
          ttile: ttile.to_string(),
          op_name: toperator.to_string(),
          lhs,
          rhs,
        }
      }
    }

    "concatenate" => {
      // concatenate(a, b), dimensions={0}

      // Inputs are comma-separated in operands, remove braces if any
      let inputs = toperands
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();

      // The one dimension it joins along, dimensions={0}
      let text = required(tsymbol, toperator, &attributes, "dimensions")?;
      let dimensions = match text.parse::<u32>() {
        Ok(dimension) => vec![dimension],
        Err(_) => {
          return Err(format!(
            "{} concatenates along {{{}}}, where a concatenate takes one dimension",
            tsymbol, text
          ))
        }
      };

      HLOExpr::Concatenate {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        tinputs: inputs,
        dimensions,
      }
    }

    "reshape" => {
      // reshape(Arg_0.6)
      let source = toperands.trim().to_string();

      HLOExpr::Reshape {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      }
    }

    "transpose" => {
      let dim_str = required(tsymbol, toperator, &attributes, "dimensions")?;
      let dimensions = dimension_list(tsymbol, "dimensions", dim_str)?;

      let source = toperands.trim().to_string();

      HLOExpr::Transpose {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        dimensions: shape_text(&dimensions),
      }
    }

    "convert" => {
      let source = toperands.trim().to_string();
      HLOExpr::Convert {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      }
    }

    "reduce" => {
      // The dimensions it sums over, dimensions={0,1}, and its region
      let dim_str = required(tsymbol, toperator, &attributes, "dimensions")?;
      let dimensions = shape_text(&dimension_list(tsymbol, "dimensions", dim_str)?);
      let to_apply = required(tsymbol, toperator, &attributes, "to_apply")?.to_string();

      validate_region_exists(&to_apply, regions)?;

      // Parse the operands to separate source and constant
      let operands: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
      if operands.len() != 2 {
        return Err(format!(
          "{} reduces {} operands, where a reduce takes 2, a value and its initial value",
          tsymbol,
          operands.len()
        ));
      }

      let source = operands[0].to_string();
      let init = operands[1].to_string();

      HLOExpr::ReduceSum {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        init,
        dimensions,
      }
    }
    "exponential" => {
      let source = toperands.trim().to_string();
      HLOExpr::Exponential {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      }
    }
    _ => {
      return Err(format!(
        "{} applies {}, an operation the backend does not support",
        tsymbol, toperator
      ));
    }
  };

  Ok(expr)
}
