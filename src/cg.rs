//! `cg` module provides basic `Vector2` `Vector3` struct with primitive scalars and also provides
//! FPN inegrated version. Common `ops` traits are implemented here.

use crate::base::FPN;
use crate::common::F64;
use std::ops::*;
use core::cmp;
use typenum::*;

pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub struct Math {}

impl Math {
    pub fn pi() -> F64 {
        F64::load(PIS12)
    }

    pub fn pi_double() -> F64 {
        F64::load(PIS12_DOUBLE)
    }

    pub fn pi_square() -> F64 {
        F64::load(PIS12_SQUARE)
    }
}

pub type FVector2<I, F> = Vector2<FPN<I, F>>;
pub type FVector3<I, F> = Vector3<FPN<I, F>>;
pub type F64Vector2 = FVector2<i64, U12>;
pub type F64Vector3 = FVector3<i64, U12>;

impl<T> Clone for Vector2<T> where T: Clone {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
        }
    }
}

impl<T> Clone for Vector3<T> where T: Clone {
    fn clone(&self) -> Self {
        Self {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }
}

impl<T> cmp::PartialEq for Vector2<T> where T: cmp::PartialEq {
    fn eq(&self, v: &Self) -> bool {
        self.x.eq(&v.x) && self.y.eq(&v.y)
    }
}

impl<T> cmp::Eq for Vector2<T> where T: cmp::Eq { }

impl<T> cmp::PartialEq for Vector3<T> where T: cmp::PartialEq {
    fn eq(&self, v: &Self) -> bool {
        self.x.eq(&v.x) && self.y.eq(&v.y) && self.z.eq(&v.z)
    }
}

impl<T> cmp::Eq for Vector3<T> where T: cmp::Eq { }

impl<T> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

pub trait Dot<T> {
    fn dot(&self, v: Self) -> T;
}

pub trait Cross<T> {
    fn cross(&self, v: Self) -> Self;
}

/// Convert from cartesian cooridates to polar coordinates
/// For Vector2 return Vector2 x: distance, y: couterclock angle from X - axis
/// For Vector3 return Vector3 x: distance, y: couterclock angle from X - axis, z: couterclock angle from
/// Y - axis
pub trait Polar<T> {
    fn polar(&self) -> Self;
}

macro_rules! impl_cg2_ops {
    ($($ty: ty),+) => {
        $(
            impl Dot<$ty> for Vector2<$ty> {
                fn dot(&self, v: Self) -> $ty {
                    self.x * v.x + self.y * v.y
                }
            }

            impl Polar<Vector2<$ty>> for Vector2<$ty> {
                fn polar(&self) -> Self {
                    self.clone()
                }
            }

            impl Neg for Vector2<$ty> {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl Neg for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl Add for Vector2<$ty> {
                type Output = Self;
                fn add(self, v: Self) -> Self {
                    Self {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl Add<Vector2<$ty>> for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn add(self, v: Vector2<$ty>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl Add for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl Add<&Vector2<$ty>> for Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl AddAssign for Vector2<$ty> {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl AddAssign<&Vector2<$ty>> for Vector2<$ty> {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl Sub for Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl Sub for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl Sub<&Vector2<$ty>> for Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl Sub<Vector2<$ty>> for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn sub(self, v: Vector2<$ty>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl SubAssign for Vector2<$ty> {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl SubAssign<&Vector2<$ty>> for Vector2<$ty> {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl Mul<$ty> for Vector2<$ty> {
                type Output = Self;
                fn mul(self, v: $ty) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl Mul<$ty> for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn mul(self, v: $ty) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl MulAssign<$ty> for Vector2<$ty> {
                fn mul_assign(&mut self, v: $ty) {
                    self.x *= v;
                    self.y *= v;
                }
            }

            impl Div<$ty> for Vector2<$ty> {
                type Output = Self;
                fn div(self, v: $ty) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl Div<$ty> for &Vector2<$ty> {
                type Output = Vector2<$ty>;
                fn div(self, v: $ty) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl DivAssign<$ty> for Vector2<$ty> {
                fn div_assign(&mut self, v: $ty) {
                    self.x /= v;
                    self.y /= v;
                }
            }
        )+
    }
}

macro_rules! impl_cg3_ops {
    ($($ty: ty),+) => {
        $(
            impl Dot<$ty> for Vector3<$ty> {
                fn dot(&self, v: Self) -> $ty {
                    self.x * v.x + self.y * v.y + self.z * v.z
                }
            }

            impl Cross<$ty> for Vector3<$ty> {
                fn cross(&self, v: Self) -> Self {
                    Self {
                        x: self.y * v.z - v.y * self.z,
                        y: self.z * v.x - v.z * self.x,
                        z: self.x * v.y - v.x * self.y,
                    }
                }
            }

            impl Neg for Vector3<$ty> {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl Neg for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl Add for Vector3<$ty> {
                type Output = Self;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl Add for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl Add<&Vector3<$ty>> for Vector3<$ty> {
                type Output = Self;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl Add<Vector3<$ty>> for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn add(self, v: Vector3<$ty>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl AddAssign for Vector3<$ty> {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl AddAssign<&Vector3<$ty>> for Vector3<$ty> {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl Sub for Vector3<$ty> {
                type Output = Self;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl Sub for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl Sub<Vector3<$ty>> for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn sub(self, v: Vector3<$ty>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }
 
            impl Sub<&Vector3<$ty>> for Vector3<$ty> {
                type Output = Self;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl SubAssign for Vector3<$ty> {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl SubAssign<&Vector3<$ty>> for Vector3<$ty> {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl Mul<$ty> for Vector3<$ty> {
                type Output = Self;
                fn mul(self, v: $ty) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl Mul<$ty> for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn mul(self, v: $ty) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl MulAssign<$ty> for Vector3<$ty> {
                fn mul_assign(&mut self, v: $ty) {
                    self.x *= v;
                    self.y *= v;
                    self.z *= v;
                }
            }

            impl Div<$ty> for Vector3<$ty> {
                type Output = Self;
                fn div(self, v: $ty) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl Div<$ty> for &Vector3<$ty> {
                type Output = Vector3<$ty>;
                fn div(self, v: $ty) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }
            impl DivAssign<$ty> for Vector3<$ty> {
                fn div_assign(&mut self, v: $ty) {
                    self.x /= v;
                    self.y /= v;
                    self.z /= v;
                }
            }
        )+
    }
}

impl_cg2_ops!(i8, i16, i32, i64, f32, f64);
impl_cg3_ops!(i8, i16, i32, i64, f32, f64);

/* Probably a bad idea here:
macro_rules! impl_cg2_ops_fpn {
    ($t:ty, $($ty: ty),+) => {
        $(
            impl_cg2_ops!(FPN<$t, $ty>);
        )+
    }
}

impl_cg2_ops_fpn!(i8, U1, U2, U3, U4, U5, U6, U7);
impl_cg2_ops_fpn!(i16, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15);
impl_cg2_ops_fpn!(i32, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31);
impl_cg2_ops_fpn!(i64, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21, U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36, U37, U38, U39, U40, U41, U42, U43, U44, U45, U46, U47, U48, U49, U50, U51, U52, U53, U54, U55, U56, U57, U58, U59, U60, U61, U62, U63);
*/

macro_rules! impl_cg2_ops_fpn_ext {
    ($($ty: ty),+) => {
        $(
            impl<F> Dot<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                fn dot(&self, v: Self) -> FPN<$ty, F> {
                    self.x * v.x + self.y * v.y
                }
            }

            impl<F> Polar<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                fn polar(&self) -> Self {
                    self.clone()
                }
            }

            impl<F> Neg for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl<F> Neg for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl<F> Add for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> Add for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> Add<&FVector2<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> Add<FVector2<$ty, F>> for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn add(self, v: FVector2<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> AddAssign for FVector2<$ty, F> where F: Unsigned {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl<F> AddAssign<&FVector2<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl<F> Sub for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> Sub for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> Sub<&Self> for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> Sub<FVector2<$ty, F>> for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn sub(self, v: FVector2<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> SubAssign for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl<F> SubAssign<&Self> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl<F> Mul<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: FPN<$ty, F>) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl<F> Mul<FPN<$ty, F>> for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn mul(self, v: FPN<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl<F> MulAssign<FPN<$ty, F>> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn mul_assign(&mut self, v: FPN<$ty, F>) {
                    self.x *= v;
                    self.y *= v;
                }
            }

            impl<F> Div<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: FPN<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl<F> Div<FPN<$ty, F>> for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn div(self, v: FPN<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl<F> DivAssign<FPN<$ty, F>> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn div_assign(&mut self, v: FPN<$ty, F>) {
                    self.x /= v;
                    self.y /= v;
                }
            }
 
            impl<F> Add<Vector2<$ty>> for Vector2<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Vector2<$ty>) -> Self {
                    Self {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> AddAssign<Vector2<$ty>> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn add_assign(&mut self, v: Vector2<$ty>) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl<F> Sub<Vector2<$ty>> for Vector2<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Vector2<$ty>) -> Self {
                    Self {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> SubAssign<Vector2<$ty>> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: Vector2<$ty>) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl<F> Mul<$ty> for Vector2<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: $ty) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl<F> MulAssign<$ty> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn mul_assign(&mut self, v: $ty) {
                    self.x *= v;
                    self.y *= v;
                }
            }

            impl<F> Div<$ty> for Vector2<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn div(self, v: $ty) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl<F> DivAssign<$ty> for Vector2<FPN<$ty, F>> where F: Unsigned {
                fn div_assign(&mut self, v: $ty) {
                    self.x /= v;
                    self.y /= v;
                }
            }

            impl<F> Shr<u8> for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                    }
                }
            }

            impl<F> Shr<u8> for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                    }
                }
            }

            impl<F> ShrAssign<u8> for FVector2<$ty, F> where F: Unsigned {
                fn shr_assign(&mut self, v: u8) {
                    self.x >>= v;
                    self.y >>= v;
                }
            }

            impl<F> Shl<u8> for FVector2<$ty, F> where F: Unsigned {
                type Output = Self;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                    }
                }
            }

            impl<F> Shl<u8> for &FVector2<$ty, F> where F: Unsigned {
                type Output = FVector2<$ty, F>;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                    }
                }
            }

            impl<F> ShlAssign<u8> for FVector2<$ty, F> where F: Unsigned {
                fn shl_assign(&mut self, v: u8) {
                    self.x <<= v;
                    self.y <<= v;
                }
            }
        )+
    }
}

macro_rules! impl_cg3_ops_fpn_ext {
    ($($ty: ty),+) => {
        $(
            impl<F> Dot<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                fn dot(&self, v: Self) -> FPN<$ty, F> {
                    self.x * v.x + self.y * v.y + self.z * v.z
                }
            }

            impl<F> Cross<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                fn cross(&self, v: Self) -> Self {
                    Self {
                        x: self.y * v.z - v.y * self.z,
                        y: self.z * v.x - v.z * self.x,
                        z: self.x * v.y - v.x * self.y,
                    }
                }
            }

            impl<F> Neg for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl<F> Neg for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl<F> Add for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> Add for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> Add<FVector3<$ty, F>> for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn add(self, v: FVector3<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> Add<&Self> for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> AddAssign for FVector3<$ty, F> where F: Unsigned {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl<F> AddAssign<&Self> for FVector3<$ty, F> where F: Unsigned {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl<F> Sub for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> Sub for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> Sub<&Self> for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> Sub<FVector3<$ty, F>> for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn sub(self, v: FVector3<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> SubAssign for FVector3<$ty, F> where F: Unsigned {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl<F> SubAssign<&Self> for FVector3<$ty, F> where F: Unsigned {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl<F> Mul<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: FPN<$ty, F>) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl<F> Mul<FPN<$ty, F>> for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty,  F>;
                fn mul(self, v: FPN<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl<F> MulAssign<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                fn mul_assign(&mut self, v: FPN<$ty, F>) {
                    self.x *= v;
                    self.y *= v;
                    self.z *= v;
                }
            }

            impl<F> Div<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: FPN<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl<F> Div<FPN<$ty, F>> for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn div(self, v: FPN<$ty, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl<F> DivAssign<FPN<$ty, F>> for Vector3<FPN<$ty, F>> where F: Unsigned {
                fn div_assign(&mut self, v: FPN<$ty, F>) {
                    self.x /= v;
                    self.y /= v;
                    self.z /= v;
                }
            }
 
            impl<F> Add<Vector3<$ty>> for Vector3<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Vector3<$ty>) -> Self {
                    Self {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> AddAssign<Vector3<$ty>> for Vector3<FPN<$ty, F>> where F: Unsigned {
                fn add_assign(&mut self, v: Vector3<$ty>) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl<F> Sub<Vector3<$ty>> for Vector3<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Vector3<$ty>) -> Self {
                    Self {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> SubAssign<Vector3<$ty>> for Vector3<FPN<$ty, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: Vector3<$ty>) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl<F> Mul<$ty> for Vector3<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: $ty) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl<F> MulAssign<$ty> for Vector3<FPN<$ty, F>> where F: Unsigned {
                fn mul_assign(&mut self, v: $ty) {
                    self.x *= v;
                    self.y *= v;
                    self.z *= v;
                }
            }

            impl<F> Div<$ty> for Vector3<FPN<$ty, F>> where F: Unsigned {
                type Output = Self;
                fn div(self, v: $ty) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl<F> DivAssign<$ty> for Vector3<FPN<$ty, F>> where F: Unsigned {
                fn div_assign(&mut self, v: $ty) {
                    self.x /= v;
                    self.y /= v;
                    self.z /= v;
                }
            }

            impl<F> Shr<u8> for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                        z: self.z >> v,
                    }
                }
            }

            impl<F> Shr<u8> for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                        z: self.z >> v,
                    }
                }
            }

            impl<F> ShrAssign<u8> for FVector3<$ty, F> where F: Unsigned {
                fn shr_assign(&mut self, v: u8) {
                    self.x >>= v;
                    self.y >>= v;
                    self.z >>= v;
                }
            }

            impl<F> Shl<u8> for FVector3<$ty, F> where F: Unsigned {
                type Output = Self;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                        z: self.z << v,
                    }
                }
            }

            impl<F> Shl<u8> for &FVector3<$ty, F> where F: Unsigned {
                type Output = FVector3<$ty, F>;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                        z: self.z << v,
                    }
                }
            }

            impl<F> ShlAssign<u8> for FVector3<$ty, F> where F: Unsigned {
                fn shl_assign(&mut self, v: u8) {
                    self.x <<= v;
                    self.y <<= v;
                    self.z <<= v;
                }
            }

        )+
    }
}

impl_cg2_ops_fpn_ext!(i8, i16, i32, i64);
impl_cg3_ops_fpn_ext!(i8, i16, i32, i64);

#[macro_export]
macro_rules! fv2_eq {
    ($a: expr, $b: expr) => {
        fv2_eq!($a, $b, $a.x.get_eps());
    };
    ($a: expr, $b: expr, $eps: expr) => {
        eq_with_eps!($a.x, $b.x, $eps);
        eq_with_eps!($a.y, $b.y, $eps);
    }
}

#[macro_export]
macro_rules! fv3_eq {
    ($a: expr, $b: expr) => {
        fv3_eq!($a, $b, $a.x.get_eps());
    };
    ($a: expr, $b: expr, $eps: expr) => {
        eq_with_eps!($a.x, $b.x, $eps);
        eq_with_eps!($a.y, $b.y, $eps);
        eq_with_eps!($a.z, $b.z, $eps);
    }
}

#[cfg(test)]
mod tests {
    use crate::cg::*;
    const X: f32 = 3.1415926f32;
    const Y: f32 = 39.19015926f32;
    const Z: f32 = 12.8415926f32;

    #[test]
    fn test_vector2 () {
        let eps: f32 = F64::eps();
        let v1 = Vector2::new(X, Y);
        let fv1 = F64Vector2::new(F64::new(X), F64::new(Y));
        fv2_eq!(fv1, v1);
        fv2_eq!(-&fv1, -&v1);
        fv2_eq!(&fv1 + &fv1, &v1 * 2f32, eps * 2f32);
        fv2_eq!(&fv1 << 1, &v1 * 2f32, eps * 2f32);
        fv2_eq!(&fv1 >> 1, &v1 / 2f32);
        fv2_eq!(&fv1 / F64::with(2), &v1 / 2f32);
    }

    #[test]
    fn test_vector3 () {
        let eps: f32 = F64::eps();
        let v1 = Vector3::new(X, Y, Z);
        let fv1 = F64Vector3::new(F64::new(X), F64::new(Y), F64::new(Z));
        fv3_eq!(fv1, v1);
        fv3_eq!(-&fv1, -&v1);
        fv3_eq!(&fv1 + &fv1, &v1 * 2f32, eps * 2f32);
        fv3_eq!(&fv1 << 1, &v1 * 2f32, eps * 2f32);
        fv3_eq!(&fv1 >> 1, &v1 / 2f32);
        fv3_eq!(&fv1 / F64::with(2), &v1 / 2f32);
    }
}

/// Constants used to convert values from cartesian coordinates to polar coordinates
/// Constant angles and tan(a) from a = (PI/4) * 1 / 90, tan(a) to a = (PI/4) * 90 / 90, tan(a)
const ANGLES: [i64; 90] = [
    36i64,
    71i64,
    107i64,
    143i64,
    179i64,
    214i64,
    250i64,
    286i64,
    322i64,
    357i64,
    393i64,
    429i64,
    465i64,
    500i64,
    536i64,
    572i64,
    608i64,
    643i64,
    679i64,
    715i64,
    751i64,
    786i64,
    822i64,
    858i64,
    894i64,
    929i64,
    965i64,
    1001i64,
    1037i64,
    1072i64,
    1108i64,
    1144i64,
    1180i64,
    1215i64,
    1251i64,
    1287i64,
    1323i64,
    1358i64,
    1394i64,
    1430i64,
    1466i64,
    1501i64,
    1537i64,
    1573i64,
    1608i64,
    1644i64,
    1680i64,
    1716i64,
    1751i64,
    1787i64,
    1823i64,
    1859i64,
    1894i64,
    1930i64,
    1966i64,
    2002i64,
    2037i64,
    2073i64,
    2109i64,
    2145i64,
    2180i64,
    2216i64,
    2252i64,
    2288i64,
    2323i64,
    2359i64,
    2395i64,
    2431i64,
    2466i64,
    2502i64,
    2538i64,
    2574i64,
    2609i64,
    2645i64,
    2681i64,
    2717i64,
    2752i64,
    2788i64,
    2824i64,
    2860i64,
    2895i64,
    2931i64,
    2967i64,
    3003i64,
    3038i64,
    3074i64,
    3110i64,
    3146i64,
    3181i64,
    3217i64,
];
const TANS: [i64; 90] = [
    36i64,
    71i64,
    107i64,
    143i64,
    179i64,
    215i64,
    251i64,
    286i64,
    322i64,
    358i64,
    394i64,
    431i64,
    467i64,
    503i64,
    539i64,
    576i64,
    612i64,
    649i64,
    685i64,
    722i64,
    759i64,
    796i64,
    833i64,
    871i64,
    908i64,
    946i64,
    983i64,
    1021i64,
    1059i64,
    1098i64,
    1136i64,
    1175i64,
    1213i64,
    1252i64,
    1291i64,
    1331i64,
    1371i64,
    1410i64,
    1450i64,
    1491i64,
    1531i64,
    1572i64,
    1613i64,
    1655i64,
    1697i64,
    1739i64,
    1781i64,
    1824i64,
    1867i64,
    1910i64,
    1954i64,
    1998i64,
    2042i64,
    2087i64,
    2132i64,
    2178i64,
    2224i64,
    2270i64,
    2317i64,
    2365i64,
    2413i64,
    2461i64,
    2510i64,
    2559i64,
    2609i64,
    2660i64,
    2711i64,
    2763i64,
    2815i64,
    2868i64,
    2922i64,
    2976i64,
    3031i64,
    3087i64,
    3143i64,
    3200i64,
    3258i64,
    3317i64,
    3376i64,
    3437i64,
    3498i64,
    3561i64,
    3624i64,
    3688i64,
    3753i64,
    3820i64,
    3887i64,
    3955i64,
    4025i64,
    4096i64,
];
/// PI << 12, (PI ** 2) << 12, (PI * 2) << 12
const PIS12: i64 = 12868i64;
const PIS12_SQUARE: i64 = 40426i64;
const PIS12_DOUBLE: i64 = 25736i64; 
