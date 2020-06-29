//! Fixed Point Numbers
//!
//! Fixed point number is designed to provide a memory presentation for real numbers.
//! Currently implemented only for a few cases for purpose of real usages, but will make it flexible enough later.
//!
//! FPN is the struct which contains a primitive for memory presentation with a specified length of
//! bits for fractions. Normal `F64` (`FPN<i64, U12>`) and `F32` (`FPN<i32, U8>`) should be fine
//! for usages. But be careful about the range overflow which doing `Add`, `Mul` and `Div`
//!
//! `Vector2` and `Vector3` provides the graphic computation containers for 2D and 3D, while
//! `F64Vector2` (`Vector2<F64>`), `F64Vector3`, `F32Vector2` and `F32Vector3` for fixed numbers.
//!
//! `Dot` trait and `Cross` provides the dot product and cross product, while `Polar` trait
//! provides the function convert a Cartesian Coordinate to a Polar/Spherical Coordinates.

// #![feature(const_fn)]
#[macro_use] pub mod base;
#[macro_use] pub mod cg;
pub mod common;

pub use base::{ FPN, To };
pub use common::{ FPN64, FPN32, F64, F32 };

pub use cg::{
    Vector2, Vector3, FVector2, FVector3, F64Vector2, F64Vector3,
    Dot, Polar, Cross
};

