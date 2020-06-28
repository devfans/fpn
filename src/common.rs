//! Common FPNs defined including: `F64` and `F32`.

use crate::base::FPN;
use typenum::*;

pub type FPN64<F> = FPN<i64, F>;
pub type FPN32<F> = FPN<i32, F>;
pub type F64 = FPN<i64, U12>;
pub type F32 = FPN<i32, U8>;

