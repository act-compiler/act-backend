pub fn parse_type_shape(type_shape: &str) -> (String, String, String) {
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
        if let Some(start_paren) = tile_str.find('(') {
          if let Some(end_paren) = tile_str.find(')') {
            ttile = tile_str[start_paren + 1..end_paren].to_string();
          } else {
            // No closing ')'
            panic!("Invalid ttype string: {}", type_shape);
          }
        }
      }
    } else {
      // No closing ']'
      panic!("Invalid ttype string: {}", type_shape);
    }
  } else {
    // No shape
    panic!("Invalid ttype string: {}", type_shape);
  }
  if tshape.is_empty() {
    // Handle scalar types like bf16[] - they have empty shape
    tshape = "[]".to_string();
  }
  (ttype, tshape, ttile)
}

pub fn parse_line(line: &str) -> Option<(String, String, String, String, String, String, String)> {
  let line = line.trim();

  // 0. Ignore empty lines or comments
  if line.is_empty() || line.starts_with("//") {
    return None;
  }

  // 1. Split at '=' (exactly one)
  let mut parts = line.splitn(2, '=');
  let mut tsymbol = parts.next()?.trim().to_string();

  // A ROOT line, its prefix trimmed
  let is_root = tsymbol.starts_with("ROOT ");
  if is_root {
    tsymbol = tsymbol.replacen("ROOT ", "", 1).trim().to_string();
  }

  let rest = parts.next()?.trim();

  // 2. rest: type, shape, operator(operands), then the rest

  // Find the first space to get type+shape
  let first_space = rest.find(' ')?;
  let (type_shape, after_type_shape) = rest.split_at(first_space);
  let after_type_shape = after_type_shape.trim_start();

  // 3. Type, shape and tile, as in u64[3,8]{;T(16,4)}
  let (ttype, tshape, ttile) = parse_type_shape(type_shape);

  // 4. Extract operator name before '('
  let op_start = after_type_shape.find('(')?;
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

  Some((tsymbol, ttype, tshape, ttile, toperator, toperands, trest))
}
