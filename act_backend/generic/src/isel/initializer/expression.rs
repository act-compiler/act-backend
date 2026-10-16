use crate::isel::initializer::parser::HLORegion;

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
) -> bool {
  if !regions.contains_key(region_name) {
    panic!("Region {} does not exist", region_name);
  }

  // region only contains parameter and add instructions
  for instruction in regions.get(region_name).unwrap().instructions.iter() {
    match instruction {
      HLOExpr::Parameter { .. } => {}
      HLOExpr::BinaryOp { op_name, .. } if op_name == "add" => {}
      _ => {
        panic!(
          "Region {} contains non-parameter and non-add instructions",
          region_name
        );
      }
    }
  }
  true
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
) -> Option<HLOExpr> {
  match toperator {
    "parameter" => {
      // Parameter(0) -- parse the index from operands
      let tidx = toperands.trim().parse::<u32>().ok()?;
      Some(HLOExpr::Parameter {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        tidx,
      })
    }

    "constant" => {
      // constant(1002)
      let value = toperands.trim().to_string();
      Some(HLOExpr::Constant {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        value,
      })
    }

    "broadcast" => {
      // broadcast(Arg_0.6)
      let source = toperands.trim().to_string();
      Some(HLOExpr::Broadcast {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      })
    }

    "slice" => {
      // slice(source), slice={[0:1:1], [0:8:1]}: the slices in the rest
      let source = toperands.trim().to_string();

      // The text inside the braces, split at commas
      let slices = if let Some(start) = trest.find('{') {
        if let Some(end) = trest.find('}') {
          let slice_str = &trest[start + 1..end];
          slice_str
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>()
        } else {
          vec![]
        }
      } else {
        vec![]
      };

      Some(HLOExpr::Slice {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        slices,
      })
    }

    "add"
    | "xor"
    | "shift-right-logical"
    | "mul"
    | "div"
    | "and"
    | "or"
    | "shift-left"
    | "dot"
    | "divide" => {
      // binary operators with two operands separated by comma
      let operands: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
      if operands.len() != 2 {
        panic!(
          "{} expects exactly 2 operands, got {}: {}",
          toperator,
          operands.len(),
          toperands
        );
      }
      let lhs = operands[0].to_string();
      let rhs = operands[1].to_string();

      Some(HLOExpr::BinaryOp {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        op_name: toperator.to_string(),
        lhs,
        rhs,
      })
    }

    "concatenate" => {
      // concatenate({a, b, c}), dimension=0 or similar, parse inputs and dims

      // Inputs are comma-separated in operands, remove braces if any
      let inputs = toperands
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split(',')
        .map(|s| s.trim().to_string())
        .collect::<Vec<_>>();

      // Parse dimension from rest e.g. dimension=0 or dims=[0,1]
      let dimensions = trest
        .chars()
        .filter_map(|c| c.to_digit(10))
        .collect::<Vec<_>>();

      Some(HLOExpr::Concatenate {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        tinputs: inputs,
        dimensions,
      })
    }

    "reshape" => {
      // reshape(Arg_0.6), new_shape=[1, 2, 3]
      let source = toperands.trim().to_string();

      Some(HLOExpr::Reshape {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      })
    }

    "transpose" => {
      let dim_str = trest
        .split_once('{')
        .and_then(|(_, rest)| rest.split_once('}'))
        .map(|(inner, _)| inner.trim())
        .unwrap_or("");

      let source = toperands.trim().to_string();

      Some(HLOExpr::Transpose {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        dimensions: dim_str.to_string(),
      })
    }

    "convert" => {
      let source = toperands.trim().to_string();
      Some(HLOExpr::Convert {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      })
    }

    "reduce" => {
      // Parse dimensions from something like: dimensions={1}
      let dimensions = trest
        .split("dimensions=")
        .nth(1)
        .and_then(|s| s.split(',').next()) // Up to next comma or end
        .map(|s| s.trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace()))
        .unwrap_or("")
        .to_string();

      // Parse reduction operation from something like: to_apply=%region_0.10, metadata=...
      let to_apply = trest
        .split("to_apply=")
        .nth(1)
        .map(|s| {
          // Find the next comma that's not inside brackets/parentheses
          let mut depth = 0;
          let mut in_brackets = false;
          let mut in_parens = false;

          for (i, ch) in s.char_indices() {
            match ch {
              '[' => in_brackets = true,
              ']' => in_brackets = false,
              '(' => {
                if !in_brackets {
                  in_parens = true;
                  depth += 1;
                }
              }
              ')' => {
                if !in_brackets {
                  depth -= 1;
                  if depth == 0 {
                    in_parens = false;
                  }
                }
              }
              ',' if !in_brackets && !in_parens => {
                // Found a top-level comma, take everything before it
                return s[..i].trim().to_string();
              }
              _ => {}
            }
          }
          // No comma found, take the whole string
          s.trim().to_string()
        })
        .unwrap_or("".to_string())
        .to_string();

      validate_region_exists(&to_apply, regions);

      // Parse the operands to separate source and constant
      let operands: Vec<&str> = toperands.split(',').map(|s| s.trim()).collect();
      if operands.len() != 2 {
        panic!(
          "Reduce operation expects exactly 2 operands, got {}: {}",
          operands.len(),
          toperands
        );
      }

      let source = operands[0].to_string();

      Some(HLOExpr::ReduceSum {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
        dimensions,
      })
    }
    "exponential" => {
      let source = toperands.trim().to_string();
      Some(HLOExpr::Exponential {
        tsymbol: tsymbol.to_string(),
        ttype: ttype.to_string(),
        tshape: tshape.to_string(),
        ttile: ttile.to_string(),
        source,
      })
    }
    _ => {
      panic!("Unknown operator: {}", toperator);
    }
  }
}
