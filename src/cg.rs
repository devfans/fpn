//! `cg` module provides basic `Vector2` `Vector3` struct with primitive scalars and also provides
//! FPN inegrated version. Common `ops` traits are implemented here.

use crate::base::{ FPN, To };
use crate::common::F64;
use std::ops::*;
use core::cmp;
use typenum::*;
use std::fmt;

/// Vector2 provides the common 2D coordinates container
/// Common `ops` traits are implemented for primitive types and FPN.
/// Vector2 is not `Copy`, so try use reference when getting borrow conflicts, like
/// ```
/// use fpn::Vector2;
///
/// let v = Vector2::new(1f32, 1f32);
/// // let added = v + v; // this will panic
/// let added = &v + &v;
/// // or
/// let double = v * 2f32;
/// ```
/// Dot product and cross product are provided with trait `Dot` and `Cross`.
/// For cross product of Vector2, `x` is the true value while `y` holds the `signum` of `x`
///
/// For `Vector2<FPN>`, bitwize shift is provided too.
///     
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

/// Vector3 provides the common 3D coordinates container
/// Common `ops` traits are implemented for primitive types and FPN.
/// Vector3 is not `Copy`, so try use reference when getting borrow conflicts, like
/// ```
/// use fpn::Vector3;
///
/// let v = Vector3::new(1f32, 1f32, 1f32);
/// // let added = v + v; // this will panic
/// let added = &v + &v;
/// // or
/// let double = v * 2f32;
/// ```
/// Dot product and cross product are provided with trait `Dot` and `Cross`.
///
/// For `Vector3<FPN>`, bitwize shift is provided too.
///     
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
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

impl<T> fmt::Display for Vector2<T> where T: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T> fmt::Display for Vector3<T> where T: fmt::Display {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl<T> fmt::Debug for Vector2<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector2")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<T> fmt::Debug for Vector3<T> where T: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vector3")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
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
    fn dot(&self, v: &Self) -> T;
}

pub trait Cross<T> {
    fn cross(&self, v: &Self) -> Self;
}

/// Convert from cartesian cooridates to polar/sphere coordinates
///
/// For Vector2 return Vector2 x: distance, y: couterclock angle from X-axix
/// For Vector3 return Vector3 x: distance, y: couterclock angle from X-axis, z: angle from
/// positive Y-axis
/// 
/// If T is one of the primitives, then T will be cast to f64 and use native sqrt, atan2 to
/// compute.
///
/// If T is FPN, then T will be converted to `F64` using trait `To`. The computation is using
/// binary search against a prepared angle array(from `0` to `0.25pi`), with time complexity `log(K)` where
/// `K` mean the array length which is fixed as `65` currently. The result difference for radius
/// distance is about `radius * 0.25pi/65` which is about `0.012 * radius`, for angle the
/// difference is about `0.012` plus the `eps()` of the specific `FPN`
pub trait Polar<T> {
    fn polar(&self) -> Self;
    fn distance(&self) -> T;
    fn distance_square(&self) -> T;
}


macro_rules! impl_cg2_ops {
    ($($ty: ty),+) => {
        $(
            impl Dot<$ty> for Vector2<$ty> {
                fn dot(&self, v: &Self) -> $ty {
                    self.x * v.x + self.y * v.y
                }
            }

            impl Cross<$ty> for Vector2<$ty> {
                fn cross(&self, v: &Self) -> Self {
                    let x = self.x * v.y - v.x * self.y;
                    Self {
                        x,
                        y: x.signum()
                    }
                }
            }

            impl Polar<$ty> for Vector2<$ty> {
                fn polar(&self) -> Self {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    Self {
                        x: (x * x + y * y).sqrt() as $ty,
                        y: y.atan2(x) as $ty,
                    }
                }

                fn distance(&self) -> $ty {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    (x * x + y * y).sqrt() as $ty
                }

                fn distance_square(&self) -> $ty {
                    self.x * self.x + self.y * self.y
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
                fn dot(&self, v: &Self) -> $ty {
                    self.x * v.x + self.y * v.y + self.z * v.z
                }
            }

            impl Polar<$ty> for Vector3<$ty> {
                fn polar(&self) -> Self {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    let z = self.z as f64;
                    let r = (x.powf(2f64) + y.powf(2f64) + z.powf(2f64)).sqrt();
                    Self {
                        x: r as $ty,
                        y: y.atan2(x) as $ty,
                        z: (z/r).acos() as $ty,
                    }
                }

                fn distance(&self) -> $ty {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    let z = self.z as f64;
                    (x.powf(2f64) + y.powf(2f64) + z.powf(2f64)).sqrt() as $ty
                }

                fn distance_square(&self) -> $ty {
                    self.x * self.x + self.y * self.y + self.z * self.z
                }
            }

            impl Cross<$ty> for Vector3<$ty> {
                fn cross(&self, v: &Self) -> Self {
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

pub fn f64_polar(v: &F64Vector2) -> F64Vector2 {
    macro_rules! watch {
        ($v: expr, $incr: expr, $s: expr, $d: expr) => {
            {
                let mut l  = 0;
		let length = POLAR_TANS.len() - 1;
                let mut step = length >> 1;
                let mut i = l + step;
                let target = $v.abs();
                let mut d;
                while step > 0 {
                    d = $incr;
                    d.mul_raw(POLAR_TANS[i]);
                    match (d - target).signum() {
                        0 => {
                            break;
                        },
                        1 => {
                            // Sig flipped
                        },
                        -1 => {
                            // Sig not flipped
                            l = i;
                        },
                        _ => panic!("Unexpected signum")
                    }
                    step >>= 1;
                    i = l + step;
                }
                F64Vector2 {
                    x: $incr / F64::load(POLAR_COSS[i]),
                    y: $d + ((F64::pi_quad() * i as i64) / (length as i64)) * $s,
                }
            }
        }
    }
    match (v.x.signum(), v.y.signum(), v.x.abs() > v.y.abs()) {
        (0, 0, _) => {
            F64Vector2 {
                x: F64::zero(),
                y: F64::zero()
            }
        },
        (a, 0, _) => {
            F64Vector2 {
                x: v.x.abs(),
                y: if a > 0 { F64::zero() } else { F64::pi() }
            }
        },
        (0, b, _) => {
            F64Vector2 {
                x: v.y.abs(),
                y: if b < 0 { F64::pi_half() } else { 
                    F64::pi() + F64::pi_half()
                }
            }
        },
        (1, 1, true) => {
            watch!(v.y, v.x.abs(), 1, F64::zero())
        },
        (1, 1, false) => {
            watch!(v.x, v.y.abs(), -1, F64::pi_half())
        },
        (-1, 1, true) => {
            watch!(v.y, v.x.abs(), -1, F64::pi())
        },
        (-1, 1, false) => {
            watch!(v.x, v.y.abs(), 1, F64::pi_half())
        },
        (1, -1, true) => {
            watch!(v.y, v.x.abs(), -1, F64::pi_double())
        },
        (1, -1, false) => {
            watch!(v.x, v.y.abs(), 1, F64::pi_half() + F64::pi())
        },
        (-1, -1, true) => {
            watch!(v.y, v.x.abs(), 1, F64::pi())
        },
        (-1, -1, false) => {
            watch!(v.x, v.y.abs(), -1, F64::pi_half() + F64::pi())
        }
        _ => panic!("Unexpected signum")
    }
}

pub fn f64_sphere(v: &F64Vector3) -> F64Vector3 {
    // X-Y plane first
    let xy = F64Vector2::new(v.x, v.y).polar();
    // Cast to X-Z plane
    let xz = F64Vector2::new(v.z.abs(), xy.x).polar();
    let z = if v.z.is_positive() {
        xz.y
    } else {
        F64::pi() - xz.y
    };
    F64Vector3::new(xz.x, xy.y, z)
}

macro_rules! impl_cg2_ops_fpn_ext {
    ($($ty: ty),+) => {
        $(
            impl<F> Dot<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                fn dot(&self, v: &Self) -> FPN<$ty, F> {
                    self.x * v.x + self.y * v.y
                }
            }

            impl<F> Cross<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                fn cross(&self, v: &Self) -> Self {
                    let x = self.x * v.y - v.x * self.y;
                    Self {
                        x,
                        y: FPN::<$ty, F>::with(x.signum())
                    }
                }
            }

            impl<F> Polar<FPN<$ty, F>> for FVector2<$ty, F> where F: Unsigned {
                fn polar(&self) -> Self {
                    let fv = f64_polar(&F64Vector2::new(self.x.to(), self.y.to()));
                    Self {
                        x: fv.x.to(),
                        y: fv.y.to(),
                    }
                }

                fn distance(&self) -> FPN<$ty, F> {
                    self.polar().x
                }

                fn distance_square(&self) -> FPN<$ty, F> {
                    self.x * self.x + self.y * self.y
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
                fn dot(&self, v: &Self) -> FPN<$ty, F> {
                    self.x * v.x + self.y * v.y + self.z * v.z
                }
            }

            impl<F> Polar<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                fn polar(&self) -> Self {
                    let fv = f64_sphere(&F64Vector3::new(self.x.to(), self.y.to(), self.z.to()));
                    Self {
                        x: fv.x.to(),
                        y: fv.y.to(),
                        z: fv.z.to(),
                    }
                }

                fn distance(&self) -> FPN<$ty, F> {
                    self.polar().x
                }

                fn distance_square(&self) -> FPN<$ty, F> {
                    self.x * self.x + self.y * self.y + self.z * self.z
                }
            }

            impl<F> Cross<FPN<$ty, F>> for FVector3<$ty, F> where F: Unsigned {
                fn cross(&self, v: &Self) -> Self {
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
    fn test_polar2 () {
        let v1 = Vector2::new(X, Y).polar();
        let fv1 = F64Vector2::new(F64::new(X), F64::new(Y)).polar();
        eq_with_eps!(fv1.x, v1.x, v1.x * 0.13);
        eq_with_eps!(fv1.y, v1.y, 0.13 + F64::eps());
    }

    #[test]
    fn test_polar3 () {
        let v1 = Vector3::new(X, Y, Z).polar();
        let fv1 = F64Vector3::new(F64::new(X), F64::new(Y), F64::new(Z)).polar();
        eq_with_eps!(fv1.x, v1.x, v1.z * v1.x * 0.13);
        eq_with_eps!(fv1.y, v1.y, 0.13 + F64::eps());
        eq_with_eps!(fv1.z, v1.z, 0.13 + 2f32 * F64::eps());
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
/// Constant tan(a) and cos(a) values with a from (PI/4) * 0 / 64 to (PI/4) * 64 / 64
const POLAR_TANS: [i64; 65] = [
    0i64,
    50i64,
    101i64,
    151i64,
    201i64,
    252i64,
    302i64,
    353i64,
    403i64,
    454i64,
    505i64,
    556i64,
    608i64,
    659i64,
    711i64,
    763i64,
    815i64,
    867i64,
    920i64,
    973i64,
    1026i64,
    1080i64,
    1134i64,
    1188i64,
    1243i64,
    1298i64,
    1353i64,
    1409i64,
    1466i64,
    1523i64,
    1580i64,
    1638i64,
    1697i64,
    1756i64,
    1816i64,
    1876i64,
    1937i64,
    1999i64,
    2062i64,
    2125i64,
    2189i64,
    2254i64,
    2320i64,
    2387i64,
    2455i64,
    2524i64,
    2594i64,
    2665i64,
    2737i64,
    2810i64,
    2885i64,
    2961i64,
    3038i64,
    3116i64,
    3197i64,
    3278i64,
    3362i64,
    3446i64,
    3533i64,
    3622i64,
    3712i64,
    3805i64,
    3900i64,
    3997i64,
    4096i64,
];
const POLAR_COSS: [i64; 65] = [
    4096i64,
    4096i64,
    4095i64,
    4093i64,
    4091i64,
    4088i64,
    4085i64,
    4081i64,
    4076i64,
    4071i64,
    4065i64,
    4059i64,
    4052i64,
    4044i64,
    4036i64,
    4027i64,
    4017i64,
    4007i64,
    3996i64,
    3985i64,
    3973i64,
    3961i64,
    3948i64,
    3934i64,
    3920i64,
    3905i64,
    3889i64,
    3873i64,
    3857i64,
    3839i64,
    3822i64,
    3803i64,
    3784i64,
    3765i64,
    3745i64,
    3724i64,
    3703i64,
    3681i64,
    3659i64,
    3636i64,
    3612i64,
    3588i64,
    3564i64,
    3539i64,
    3513i64,
    3487i64,
    3461i64,
    3433i64,
    3406i64,
    3378i64,
    3349i64,
    3320i64,
    3290i64,
    3260i64,
    3229i64,
    3198i64,
    3166i64,
    3134i64,
    3102i64,
    3068i64,
    3035i64,
    3001i64,
    2967i64,
    2932i64,
    2896i64,
];

