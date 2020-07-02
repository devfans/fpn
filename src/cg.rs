//! `cg` module provides basic `Vector2` `Vector3` struct with primitive scalars and also provides
//! FPN inegrated version. Common `ops` traits are implemented here.

use crate::base::{ FPN, To, TANS, COSS };
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

/// Rotate provide the coordinate conversion by rotation.
///
/// For Vector2, it rotates around the origin point in counter-clockwise direction.
pub trait Rotate2<T> {
    fn rotate(&self, v: T) -> Vector2<T>;
}

/// Rotate provide the coordinate conversion by rotation.
///
/// For Vector3, it rotates around axises in counter-clockwise direction.
pub trait Rotate3<T> {
    fn rotate_x(&self, v: T) -> Vector3<T>;
    fn rotate_y(&self, v: T) -> Vector3<T>;
    fn rotate_z(&self, v: T) -> Vector3<T>;
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
/// binary search against a prepared angle array(from `0` to `0.25pi`), with time complexity `O(log(K))` where
/// `K` mean the array length which is fixed as `65` currently. The result difference for radius
/// distance is about `radius * 0.25pi/65` which is about `0.012 * radius`, for angle the
/// difference is about `0.012` plus the `eps()` of the specific `FPN`
pub trait Polar<T> {
    fn polar(&self) -> Self;
    fn distance(&self) -> T;
    fn distance_square(&self) -> T;
}


macro_rules! impl_cg2_ops {
    ($($I: ty),+) => {
        $(
            impl Dot<$I> for Vector2<$I> {
                fn dot(&self, v: &Self) -> $I {
                    self.x * v.x + self.y * v.y
                }
            }

            impl Cross<$I> for Vector2<$I> {
                fn cross(&self, v: &Self) -> Self {
                    let x = self.x * v.y - v.x * self.y;
                    Self {
                        x,
                        y: x.signum()
                    }
                }
            }

            impl Rotate2<$I> for Vector2<$I> {
                fn rotate(&self, v: $I) -> Self {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    let angle = v as f64;
                    let i = x * angle.cos() - y * angle.sin();
                    let j = x * angle.sin() + y * angle.cos();
                    Self::new(i as $I, j as $I)
                }
            }

            impl Polar<$I> for Vector2<$I> {
                fn polar(&self) -> Self {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    Self {
                        x: (x * x + y * y).sqrt() as $I,
                        y: y.atan2(x) as $I,
                    }
                }

                fn distance(&self) -> $I {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    (x * x + y * y).sqrt() as $I
                }

                fn distance_square(&self) -> $I {
                    self.x * self.x + self.y * self.y
                }
            }

            impl Neg for Vector2<$I> {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl Neg for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl Add for Vector2<$I> {
                type Output = Self;
                fn add(self, v: Self) -> Self {
                    Self {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl Add<Vector2<$I>> for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn add(self, v: Vector2<$I>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl Add for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl Add<&Vector2<$I>> for Vector2<$I> {
                type Output = Vector2<$I>;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl AddAssign for Vector2<$I> {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl AddAssign<&Vector2<$I>> for Vector2<$I> {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl Sub for Vector2<$I> {
                type Output = Vector2<$I>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl Sub for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl Sub<&Vector2<$I>> for Vector2<$I> {
                type Output = Vector2<$I>;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl Sub<Vector2<$I>> for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn sub(self, v: Vector2<$I>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl SubAssign for Vector2<$I> {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl SubAssign<&Vector2<$I>> for Vector2<$I> {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl Mul<$I> for Vector2<$I> {
                type Output = Self;
                fn mul(self, v: $I) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl Mul<$I> for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn mul(self, v: $I) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl MulAssign<$I> for Vector2<$I> {
                fn mul_assign(&mut self, v: $I) {
                    self.x *= v;
                    self.y *= v;
                }
            }

            impl Div<$I> for Vector2<$I> {
                type Output = Self;
                fn div(self, v: $I) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl Div<$I> for &Vector2<$I> {
                type Output = Vector2<$I>;
                fn div(self, v: $I) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl DivAssign<$I> for Vector2<$I> {
                fn div_assign(&mut self, v: $I) {
                    self.x /= v;
                    self.y /= v;
                }
            }
        )+
    }
}

macro_rules! impl_cg3_ops {
    ($($I: ty),+) => {
        $(
            impl Dot<$I> for Vector3<$I> {
                fn dot(&self, v: &Self) -> $I {
                    self.x * v.x + self.y * v.y + self.z * v.z
                }
            }

            impl Polar<$I> for Vector3<$I> {
                fn polar(&self) -> Self {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    let z = self.z as f64;
                    let r = (x.powf(2f64) + y.powf(2f64) + z.powf(2f64)).sqrt();
                    Self {
                        x: r as $I,
                        y: y.atan2(x) as $I,
                        z: (z/r).acos() as $I,
                    }
                }

                fn distance(&self) -> $I {
                    let x = self.x as f64;
                    let y = self.y as f64;
                    let z = self.z as f64;
                    (x.powf(2f64) + y.powf(2f64) + z.powf(2f64)).sqrt() as $I
                }

                fn distance_square(&self) -> $I {
                    self.x * self.x + self.y * self.y + self.z * self.z
                }
            }

            impl Cross<$I> for Vector3<$I> {
                fn cross(&self, v: &Self) -> Self {
                    Self {
                        x: self.y * v.z - v.y * self.z,
                        y: self.z * v.x - v.z * self.x,
                        z: self.x * v.y - v.x * self.y,
                    }
                }
            }

            impl Rotate3<$I> for Vector3<$I> {
                fn rotate_y(&self, v: $I) -> Self {
                    let v2 = Vector2::new(self.x, self.z).rotate(v);
                    Self::new(v2.x, self.y, v2.y)
                }

                fn rotate_z(&self, v: $I) -> Self {
                    let v2 = Vector2::new(self.y, self.x).rotate(v);
                    Self::new(v2.y, v2.x, self.z)
                }

                fn rotate_x(&self, v: $I) -> Self {
                    let v2 = Vector2::new(self.z, self.y).rotate(v);
                    Self::new(self.x, v2.y, v2.x)
                }
            }

            impl Neg for Vector3<$I> {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl Neg for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl Add for Vector3<$I> {
                type Output = Self;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl Add for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl Add<&Vector3<$I>> for Vector3<$I> {
                type Output = Self;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl Add<Vector3<$I>> for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn add(self, v: Vector3<$I>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl AddAssign for Vector3<$I> {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl AddAssign<&Vector3<$I>> for Vector3<$I> {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl Sub for Vector3<$I> {
                type Output = Self;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl Sub for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl Sub<Vector3<$I>> for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn sub(self, v: Vector3<$I>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }
 
            impl Sub<&Vector3<$I>> for Vector3<$I> {
                type Output = Self;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl SubAssign for Vector3<$I> {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl SubAssign<&Vector3<$I>> for Vector3<$I> {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl Mul<$I> for Vector3<$I> {
                type Output = Self;
                fn mul(self, v: $I) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl Mul<$I> for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn mul(self, v: $I) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl MulAssign<$I> for Vector3<$I> {
                fn mul_assign(&mut self, v: $I) {
                    self.x *= v;
                    self.y *= v;
                    self.z *= v;
                }
            }

            impl Div<$I> for Vector3<$I> {
                type Output = Self;
                fn div(self, v: $I) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl Div<$I> for &Vector3<$I> {
                type Output = Vector3<$I>;
                fn div(self, v: $I) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }
            impl DivAssign<$I> for Vector3<$I> {
                fn div_assign(&mut self, v: $I) {
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
    ($t:ty, $($I: ty),+) => {
        $(
            impl_cg2_ops!(FPN<$t, $I>);
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
		let length = COSS.len() - 1;
                let mut step = length >> 1;
                let mut i = l + step;
                let target = $v.abs();
                let mut d;
                while step > 0 {
                    d = $incr;
                    d.mul_raw(TANS[i]);
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
                    x: $incr / F64::load(COSS[i]),
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
    ($($I: ty),+) => {
        $(
            impl<F> FVector2<$I, F> where F: Unsigned {
                pub fn with(x: $I, y: $I) -> Self {
                    Self {
                        x: FPN::<$I, F>::with(x),
                        y: FPN::<$I, F>::with(y),
                    }
                }

                pub fn set_x(&mut self, v: $I) {
                    self.x = FPN::<$I, F>::with(v);
                }

                pub fn set_y(&mut self, v: $I) {
                    self.y = FPN::<$I, F>::with(v);
                }
            }

            impl<F> Dot<FPN<$I, F>> for FVector2<$I, F> where F: Unsigned {
                fn dot(&self, v: &Self) -> FPN<$I, F> {
                    self.x * v.x + self.y * v.y
                }
            }

            impl<F> Cross<FPN<$I, F>> for FVector2<$I, F> where F: Unsigned {
                fn cross(&self, v: &Self) -> Self {
                    let x = self.x * v.y - v.x * self.y;
                    Self {
                        x,
                        y: FPN::<$I, F>::with(x.signum())
                    }
                }
            }

            impl<F> Rotate2<FPN<$I, F>> for FVector2<$I, F> where F: Unsigned {
                fn rotate(&self, v: FPN<$I, F>) -> Self {
                    let x: F64 = self.x.to();
                    let y: F64 = self.y.to();
                    let angle: F64 = v.to();
                    let i = x * angle.cos() - y * angle.sin();
                    let j = x * angle.sin() + y * angle.cos();
                    Self::new(i.to(), j.to())
                }
            }

            impl<F> Polar<FPN<$I, F>> for FVector2<$I, F> where F: Unsigned {
                fn polar(&self) -> Self {
                    let fv = f64_polar(&F64Vector2::new(self.x.to(), self.y.to()));
                    Self {
                        x: fv.x.to(),
                        y: fv.y.to(),
                    }
                }

                fn distance(&self) -> FPN<$I, F> {
                    self.polar().x
                }

                fn distance_square(&self) -> FPN<$I, F> {
                    self.x * self.x + self.y * self.y
                }
            }

            impl<F> Neg for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl<F> Neg for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                    }
                }
            }

            impl<F> Add for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> Add for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> Add<&FVector2<$I, F>> for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> Add<FVector2<$I, F>> for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn add(self, v: FVector2<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> AddAssign for FVector2<$I, F> where F: Unsigned {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl<F> AddAssign<&FVector2<$I, F>> for FVector2<$I, F> where F: Unsigned {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl<F> Sub for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> Sub for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> Sub<&Self> for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> Sub<FVector2<$I, F>> for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn sub(self, v: FVector2<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> SubAssign for Vector2<FPN<$I, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl<F> SubAssign<&Self> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl<F> Mul<FPN<$I, F>> for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: FPN<$I, F>) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl<F> Mul<FPN<$I, F>> for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn mul(self, v: FPN<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl<F> MulAssign<FPN<$I, F>> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn mul_assign(&mut self, v: FPN<$I, F>) {
                    self.x *= v;
                    self.y *= v;
                }
            }

            impl<F> Div<FPN<$I, F>> for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: FPN<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl<F> Div<FPN<$I, F>> for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn div(self, v: FPN<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl<F> DivAssign<FPN<$I, F>> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn div_assign(&mut self, v: FPN<$I, F>) {
                    self.x /= v;
                    self.y /= v;
                }
            }
 
            impl<F> Add<Vector2<$I>> for Vector2<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Vector2<$I>) -> Self {
                    Self {
                        x: self.x + v.x,
                        y: self.y + v.y,
                    }
                }
            }

            impl<F> AddAssign<Vector2<$I>> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn add_assign(&mut self, v: Vector2<$I>) {
                    self.x += v.x;
                    self.y += v.y;
                }
            }

            impl<F> Sub<Vector2<$I>> for Vector2<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Vector2<$I>) -> Self {
                    Self {
                        x: self.x - v.x,
                        y: self.y - v.y,
                    }
                }
            }

            impl<F> SubAssign<Vector2<$I>> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: Vector2<$I>) {
                    self.x -= v.x;
                    self.y -= v.y;
                }
            }

            impl<F> Mul<$I> for Vector2<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: $I) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                    }
                }
            }

            impl<F> MulAssign<$I> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn mul_assign(&mut self, v: $I) {
                    self.x *= v;
                    self.y *= v;
                }
            }

            impl<F> Div<$I> for Vector2<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn div(self, v: $I) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                    }
                }
            }

            impl<F> DivAssign<$I> for Vector2<FPN<$I, F>> where F: Unsigned {
                fn div_assign(&mut self, v: $I) {
                    self.x /= v;
                    self.y /= v;
                }
            }

            impl<F> Shr<u8> for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                    }
                }
            }

            impl<F> Shr<u8> for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                    }
                }
            }

            impl<F> ShrAssign<u8> for FVector2<$I, F> where F: Unsigned {
                fn shr_assign(&mut self, v: u8) {
                    self.x >>= v;
                    self.y >>= v;
                }
            }

            impl<F> Shl<u8> for FVector2<$I, F> where F: Unsigned {
                type Output = Self;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                    }
                }
            }

            impl<F> Shl<u8> for &FVector2<$I, F> where F: Unsigned {
                type Output = FVector2<$I, F>;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                    }
                }
            }

            impl<F> ShlAssign<u8> for FVector2<$I, F> where F: Unsigned {
                fn shl_assign(&mut self, v: u8) {
                    self.x <<= v;
                    self.y <<= v;
                }
            }
        )+
    }
}

macro_rules! impl_cg3_ops_fpn_ext {
    ($($I: ty),+) => {
        $(
            impl<F> FVector3<$I, F> where F: Unsigned {
                pub fn with(x: $I, y: $I, z: $I) -> Self {
                    Self {
                        x: FPN::<$I, F>::with(x),
                        y: FPN::<$I, F>::with(y),
                        z: FPN::<$I, F>::with(z),
                    }
                }

                pub fn set_x(&mut self, v: $I) {
                    self.x = FPN::<$I, F>::with(v);
                }

                pub fn set_y(&mut self, v: $I) {
                    self.y = FPN::<$I, F>::with(v);
                }

                pub fn set_z(&mut self, v: $I) {
                    self.z = FPN::<$I, F>::with(v);
                }
            }

            impl<F> Dot<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                fn dot(&self, v: &Self) -> FPN<$I, F> {
                    self.x * v.x + self.y * v.y + self.z * v.z
                }
            }

            impl<F> Polar<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                fn polar(&self) -> Self {
                    let fv = f64_sphere(&F64Vector3::new(self.x.to(), self.y.to(), self.z.to()));
                    Self {
                        x: fv.x.to(),
                        y: fv.y.to(),
                        z: fv.z.to(),
                    }
                }

                fn distance(&self) -> FPN<$I, F> {
                    self.polar().x
                }

                fn distance_square(&self) -> FPN<$I, F> {
                    self.x * self.x + self.y * self.y + self.z * self.z
                }
            }

            impl<F> Cross<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                fn cross(&self, v: &Self) -> Self {
                    Self {
                        x: self.y * v.z - v.y * self.z,
                        y: self.z * v.x - v.z * self.x,
                        z: self.x * v.y - v.x * self.y,
                    }
                }
            }

            impl<F> Rotate3<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                fn rotate_y(&self, v: FPN<$I, F>) -> Self {
                    let v2 = Vector2::new(self.x, self.z).rotate(v);
                    Self::new(v2.x, self.y, v2.y)
                }

                fn rotate_z(&self, v: FPN<$I, F>) -> Self {
                    let v2 = Vector2::new(self.y, self.x).rotate(v);
                    Self::new(v2.y, v2.x, self.z)
                }

                fn rotate_x(&self, v: FPN<$I,F>) -> Self {
                    let v2 = Vector2::new(self.z, self.y).rotate(v);
                    Self::new(self.x, v2.y, v2.x)
                }
            }

            impl<F> Neg for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl<F> Neg for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        x: self.x.neg(),
                        y: self.y.neg(),
                        z: self.z.neg(),
                    }
                }
            }

            impl<F> Add for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> Add for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn add(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> Add<FVector3<$I, F>> for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn add(self, v: FVector3<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> Add<&Self> for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> AddAssign for FVector3<$I, F> where F: Unsigned {
                fn add_assign(&mut self, v: Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl<F> AddAssign<&Self> for FVector3<$I, F> where F: Unsigned {
                fn add_assign(&mut self, v: &Self) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl<F> Sub for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> Sub for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn sub(self, v: Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> Sub<&Self> for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: &Self) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> Sub<FVector3<$I, F>> for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn sub(self, v: FVector3<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> SubAssign for FVector3<$I, F> where F: Unsigned {
                fn sub_assign(&mut self, v: Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl<F> SubAssign<&Self> for FVector3<$I, F> where F: Unsigned {
                fn sub_assign(&mut self, v: &Self) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl<F> Mul<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: FPN<$I, F>) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl<F> Mul<FPN<$I, F>> for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I,  F>;
                fn mul(self, v: FPN<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl<F> MulAssign<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                fn mul_assign(&mut self, v: FPN<$I, F>) {
                    self.x *= v;
                    self.y *= v;
                    self.z *= v;
                }
            }

            impl<F> Div<FPN<$I, F>> for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: FPN<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl<F> Div<FPN<$I, F>> for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn div(self, v: FPN<$I, F>) -> Self::Output {
                    Self::Output {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl<F> DivAssign<FPN<$I, F>> for Vector3<FPN<$I, F>> where F: Unsigned {
                fn div_assign(&mut self, v: FPN<$I, F>) {
                    self.x /= v;
                    self.y /= v;
                    self.z /= v;
                }
            }
 
            impl<F> Add<Vector3<$I>> for Vector3<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Vector3<$I>) -> Self {
                    Self {
                        x: self.x + v.x,
                        y: self.y + v.y,
                        z: self.z + v.z,
                    }
                }
            }

            impl<F> AddAssign<Vector3<$I>> for Vector3<FPN<$I, F>> where F: Unsigned {
                fn add_assign(&mut self, v: Vector3<$I>) {
                    self.x += v.x;
                    self.y += v.y;
                    self.z += v.z;
                }
            }

            impl<F> Sub<Vector3<$I>> for Vector3<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Vector3<$I>) -> Self {
                    Self {
                        x: self.x - v.x,
                        y: self.y - v.y,
                        z: self.z - v.z,
                    }
                }
            }

            impl<F> SubAssign<Vector3<$I>> for Vector3<FPN<$I, F>> where F: Unsigned {
                fn sub_assign(&mut self, v: Vector3<$I>) {
                    self.x -= v.x;
                    self.y -= v.y;
                    self.z -= v.z;
                }
            }

            impl<F> Mul<$I> for Vector3<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: $I) -> Self {
                    Self {
                        x: self.x * v,
                        y: self.y * v,
                        z: self.z * v,
                    }
                }
            }

            impl<F> MulAssign<$I> for Vector3<FPN<$I, F>> where F: Unsigned {
                fn mul_assign(&mut self, v: $I) {
                    self.x *= v;
                    self.y *= v;
                    self.z *= v;
                }
            }

            impl<F> Div<$I> for Vector3<FPN<$I, F>> where F: Unsigned {
                type Output = Self;
                fn div(self, v: $I) -> Self {
                    Self {
                        x: self.x / v,
                        y: self.y / v,
                        z: self.z / v,
                    }
                }
            }

            impl<F> DivAssign<$I> for Vector3<FPN<$I, F>> where F: Unsigned {
                fn div_assign(&mut self, v: $I) {
                    self.x /= v;
                    self.y /= v;
                    self.z /= v;
                }
            }

            impl<F> Shr<u8> for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                        z: self.z >> v,
                    }
                }
            }

            impl<F> Shr<u8> for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn shr(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x >> v,
                        y: self.y >> v,
                        z: self.z >> v,
                    }
                }
            }

            impl<F> ShrAssign<u8> for FVector3<$I, F> where F: Unsigned {
                fn shr_assign(&mut self, v: u8) {
                    self.x >>= v;
                    self.y >>= v;
                    self.z >>= v;
                }
            }

            impl<F> Shl<u8> for FVector3<$I, F> where F: Unsigned {
                type Output = Self;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                        z: self.z << v,
                    }
                }
            }

            impl<F> Shl<u8> for &FVector3<$I, F> where F: Unsigned {
                type Output = FVector3<$I, F>;
                fn shl(self, v: u8) -> Self::Output {
                    Self::Output {
                        x: self.x << v,
                        y: self.y << v,
                        z: self.z << v,
                    }
                }
            }

            impl<F> ShlAssign<u8> for FVector3<$I, F> where F: Unsigned {
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

