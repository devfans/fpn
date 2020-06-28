//! Fixed Point Numbers
//!
//! Fixed point number is designed to provide a memory presentation for real numbers.
//! Currently implemented only for a few cases for purpose of real usages, but will make it flexible enough later.
//!
//!
//!

// #![feature(const_fn)]
#[macro_use] pub mod base;
#[macro_use] pub mod cg;
pub mod common;

pub use base::FPN;
pub use common::{ FPN64, FPN32, F64, F32 };
pub use cg::{
    Vector2, Vector3, FVector2, FVector3, F64Vector2, F64Vector3,
    Dot, Polar, Cross
};

