// A kernel line's name, type, shape, tile, operator, operands, rest
pub type Line = (String, String, String, String, String, String, String);

// A layout other than row-major {n-1,...,0}, as a reason
pub(super) fn other_than_row_major(shape: &str, layout: &str) -> Option<String> {
  let rank = shape.split(',').filter(|d| !d.trim().is_empty()).count();
  let order = layout.split(':').next().unwrap_or("").replace(' ', "");
  // A layout with no order, as a tile alone, passes
  if order.is_empty() {
    return None;
  }
  let row_major = (0..rank)
    .rev()
    .map(|d| d.to_string())
    .collect::<Vec<_>>()
    .join(",");
  (order != row_major).then(|| {
    format!(
      "the layout {{{}}} of the shape [{}] is not row-major, the only layout the backend supports",
      layout, shape
    )
  })
}

pub fn parse_type_shape(type_shape: &str) -> Result<(String, String, String), String> {
  // Why the reader cannot split a type
  let malformed = || {
    format!(
      "the type '{}' is not an element type with a shape",
      type_shape
    )
  };

  let ttype_end = type_shape.find('[').unwrap_or(type_shape.len());
  let ttype = type_shape[..ttype_end].to_string();

  let mut tshape;
  let mut ttile = String::new();
  if let Some(bracket_start) = type_shape.find('[') {
    // Find end of shape (could be before tile annotation)
    if let Some(bracket_end) = type_shape[bracket_start..].find(']') {
      let bracket_end = bracket_start + bracket_end;
      tshape = type_shape[bracket_start + 1..=bracket_end - 1].to_string();
      // Check for tile annotation after shape
      if let Some(tile_start) = type_shape[bracket_end + 1..].find('{') {
        let tile_start = bracket_end + 1 + tile_start;
        let tile_str = &type_shape[tile_start..];
        let layout = tile_str[1..].split('}').next().unwrap_or("");
        if let Some(reason) = other_than_row_major(&tshape, layout) {
          return Err(reason);
        }
        if let Some(start_paren) = tile_str.find('(') {
          if let Some(end_paren) = tile_str.find(')') {
            ttile = tile_str[start_paren + 1..end_paren].to_string();
          } else {
            // No closing ')'
            return Err(malformed());
          }
        }
      }
    } else {
      // No closing ']'
      return Err(malformed());
    }
  } else {
    // No shape
    return Err(malformed());
  }
  if tshape.is_empty() {
    // Handle scalar types like bf16[] - they have empty shape
    tshape = "[]".to_string();
  }
  Ok((ttype, tshape, ttile))
}

pub fn parse_line(line: &str) -> Result<Option<Line>, String> {
  let line = line.trim();

  // 0. Ignore empty lines or comments
  if line.is_empty() || line.starts_with("//") {
    return Ok(None);
  }

  // 1. Split at '=' (exactly one)
  let mut parts = line.splitn(2, '=');
  let mut tsymbol = parts.next().unwrap_or_default().trim().to_string();

  // A ROOT line, its prefix trimmed
  let is_root = tsymbol.starts_with("ROOT ");
  if is_root {
    tsymbol = tsymbol.replacen("ROOT ", "", 1).trim().to_string();
  }

  let Some(rest) = parts.next().map(str::trim) else {
    return Ok(None);
  };

  // 2. rest: type, shape, operator(operands), then the rest

  // Find the first space to get type+shape
  let Some(first_space) = rest.find(' ') else {
    return Ok(None);
  };
  let (type_shape, after_type_shape) = rest.split_at(first_space);
  let after_type_shape = after_type_shape.trim_start();

  // 3. Type, shape and tile, as in u64[3,8]{;T(16,4)}
  let (ttype, tshape, ttile) = parse_type_shape(type_shape)?;

  // 4. Extract operator name before '('
  let Some(op_start) = after_type_shape.find('(') else {
    return Ok(None);
  };
  let toperator = after_type_shape[..op_start].trim().to_string();

  // 5. Extract operands inside matching parentheses (find ')' after op_start)
  let operands_start = op_start + 1;
  let mut paren_count = 1;
  let mut operands_end = operands_start;

  for (i, c) in after_type_shape[operands_start..].chars().enumerate() {
    match c {
      '(' => paren_count += 1,
      ')' => {
        paren_count -= 1;
        if paren_count == 0 {
          operands_end = operands_start + i;
          break;
        }
      }
      _ => {}
    }
  }

  let toperands = after_type_shape[operands_start..operands_end]
    .trim()
    .to_string();

  // 6. Everything after the closing ')' is the rest
  let trest = after_type_shape[operands_end + 1..].trim().to_string();

  Ok(Some((
    tsymbol, ttype, tshape, ttile, toperator, toperands, trest,
  )))
}
