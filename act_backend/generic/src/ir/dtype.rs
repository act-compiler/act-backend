#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Dtype {
  Pred,
  U8,
  I8,
  U16,
  I16,
  U32,
  I32,
  U64,
  I64,
  FP16,
  FP32,
  FP64,
  BF16,
}

impl Dtype {
  pub fn size_in_bytes(&self) -> i32 {
    match self {
      Dtype::Pred => 1,
      Dtype::U8 => 1,
      Dtype::I8 => 1,
      Dtype::U16 => 2,
      Dtype::I16 => 2,
      Dtype::U32 => 4,
      Dtype::I32 => 4,
      Dtype::U64 => 8,
      Dtype::I64 => 8,
      Dtype::FP16 => 2,
      Dtype::FP32 => 4,
      Dtype::FP64 => 8,
      Dtype::BF16 => 2,
    }
  }

  // The bits a literal spells as one value of this type, else None
  pub fn bits(&self, text: &str) -> Option<u64> {
    let text = text.trim();
    match self {
      Dtype::Pred => match text {
        "false" | "0" => Some(0),
        "true" | "1" => Some(1),
        _ => None,
      },
      Dtype::U8 => text.parse::<u8>().ok().map(u64::from),
      Dtype::I8 => text.parse::<i8>().ok().map(|v| u64::from(v as u8)),
      Dtype::U16 => text.parse::<u16>().ok().map(u64::from),
      Dtype::I16 => text.parse::<i16>().ok().map(|v| u64::from(v as u16)),
      Dtype::U32 => text.parse::<u32>().ok().map(u64::from),
      Dtype::I32 => text.parse::<i32>().ok().map(|v| u64::from(v as u32)),
      Dtype::U64 => text.parse::<u64>().ok(),
      Dtype::I64 => text.parse::<i64>().ok().map(|v| v as u64),
      Dtype::FP16 => text.parse::<f32>().ok().map(f16_bits),
      Dtype::FP32 => text.parse::<f32>().ok().map(|v| u64::from(v.to_bits())),
      Dtype::FP64 => text.parse::<f64>().ok().map(f64::to_bits),
      Dtype::BF16 => text.parse::<f32>().ok().map(bf16_bits),
    }
  }
}

// An f32's bits rounded to nearest even at bf16
fn bf16_bits(value: f32) -> u64 {
  if value.is_nan() {
    return 0x7fc0;
  }
  let bits = value.to_bits();
  u64::from((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
}

// An f32's bits rounded to nearest even at f16
fn f16_bits(value: f32) -> u64 {
  let bits = value.to_bits();
  let sign = (bits >> 16) & 0x8000;
  if value.is_nan() {
    return u64::from(sign | 0x7e00);
  }
  // The exponent at f16's bias, the mantissa with its leading one
  let exponent = ((bits >> 23) & 0xff) as i32 - 112;
  let mantissa = (bits & 0x7fffff) | 0x800000;
  if exponent >= 31 {
    return u64::from(sign | 0x7c00);
  }
  // The bits dropped, more where the value is subnormal
  let shift = if exponent > 0 {
    13
  } else {
    (14 - exponent) as u32
  };
  if shift > 24 {
    return u64::from(sign);
  }
  let kept = mantissa >> shift;
  let dropped = mantissa & ((1 << shift) - 1);
  let half = 1 << (shift - 1);
  let rounded = kept + u32::from(dropped > half || (dropped == half && kept & 1 == 1));
  // A carry out of the mantissa raises the exponent
  let base = if exponent > 0 {
    (exponent as u32 - 1) << 10
  } else {
    0
  };
  u64::from(sign | (base + rounded))
}

impl std::fmt::Display for Dtype {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Dtype::Pred => write!(f, "pred"),
      Dtype::U8 => write!(f, "u8"),
      Dtype::I8 => write!(f, "s8"),
      Dtype::U16 => write!(f, "u16"),
      Dtype::I16 => write!(f, "s16"),
      Dtype::U32 => write!(f, "u32"),
      Dtype::I32 => write!(f, "s32"),
      Dtype::U64 => write!(f, "u64"),
      Dtype::I64 => write!(f, "s64"),
      Dtype::FP16 => write!(f, "f16"),
      Dtype::FP32 => write!(f, "f32"),
      Dtype::FP64 => write!(f, "f64"),
      Dtype::BF16 => write!(f, "bf16"),
    }
  }
}

impl TryFrom<&str> for Dtype {
  type Error = String;

  fn try_from(s: &str) -> Result<Self, String> {
    Ok(match s {
      "pred" => Dtype::Pred,
      "u8" => Dtype::U8,
      "s8" => Dtype::I8,
      "u16" => Dtype::U16,
      "s16" => Dtype::I16,
      "u32" => Dtype::U32,
      "s32" => Dtype::I32,
      "u64" => Dtype::U64,
      "s64" => Dtype::I64,
      "f16" => Dtype::FP16,
      "f32" => Dtype::FP32,
      "f64" => Dtype::FP64,
      "bf16" => Dtype::BF16,
      _ => {
        return Err(format!(
          "the element type '{}' is not one the backend supports",
          s
        ))
      }
    })
  }
}

impl serde::Serialize for Dtype {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: serde::Serializer,
  {
    serializer.serialize_str(match self {
      Dtype::Pred => "jnp.bool_",
      Dtype::U8 => "jnp.uint8",
      Dtype::I8 => "jnp.int8",
      Dtype::U16 => "jnp.uint16",
      Dtype::I16 => "jnp.int16",
      Dtype::U32 => "jnp.uint32",
      Dtype::I32 => "jnp.int32",
      Dtype::U64 => "jnp.uint64",
      Dtype::I64 => "jnp.int64",
      Dtype::FP16 => "jnp.float16",
      Dtype::FP32 => "jnp.float32",
      Dtype::FP64 => "jnp.float64",
      Dtype::BF16 => "jnp.bfloat16",
    })
  }
}
