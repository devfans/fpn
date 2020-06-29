//! `base` provides basic fixed point numbers and implemented common `ops` traits for FPNs with integar primitives.
//! 

use core::marker::PhantomData;
use core::cmp::{ self, Ordering };
use typenum::*;
use std::ops::*;
use std::fmt::{ self, * };

pub trait Pri {}
pub trait PriInt {}
pub trait PriFloat {}

macro_rules! impl_pri {
    ($trait: ty, $($ty: ty),+) => {
        $(
            impl $trait for $ty {}
        )+
    }
}

impl_pri!(Pri, f32, f64, u8, i8, u16, i16, u32, i32, u64, i64);
impl_pri!(PriInt, u8, i8, u16, i16, u32, i32, u64, i64);
impl_pri!(PriFloat, f32, f64);

/// I: Interge type as storage, F: number of bits for fractions.
pub struct FPN<I, F> {
    b: I,
    _m: PhantomData<F>,
}

/// `To` trait provide the way convert from `FPN<I1, F1>` to `FPN<I2, F2>`;
pub trait To<T> {
    fn to(&self) -> T;
}

impl<I, F> Clone for FPN<I, F> where I: Clone {
    fn clone(&self) -> Self { Self { b: self.b.clone(), _m: PhantomData } }
}

impl<I, F> Copy for FPN<I, F> where I: Copy { }

impl<I, F> cmp::Ord for FPN<I, F> where I: cmp::Ord {
    fn cmp(&self, rhs: &Self) -> Ordering { self.b.cmp(&rhs.b) }
}

impl<I, F> cmp::PartialOrd for FPN<I, F> where I: cmp::PartialOrd {
    fn partial_cmp(&self, v: &Self) -> Option<Ordering> { self.b.partial_cmp(&v.b) }
}

impl<I, F> cmp::PartialEq for FPN<I, F> where I: cmp::PartialEq {
    fn eq(&self, v: &Self) -> bool { self.b.eq(&v.b) }
}

impl<I, F> cmp::Eq for FPN<I, F> where I: cmp::Eq { }

/// Convert and Parse
/// Ops

macro_rules! pi {
    ($v:expr, $F: ident, $ty: ty, $Self: ident) => {
        {
            let s = $F::to_u8();
            let b = if s < 56 {
                $v >> 56 - s
            } else {
                $v << s - 56
            };
            $Self {
                b: b as $ty,
                _m: PhantomData
            }
        }
    }
}

macro_rules! impl_ops {
    ($($ty: ty),+) => {
        $(
            impl<F> FPN<$ty, F> where F: Unsigned {
                pub fn pi() -> Self {
                    pi!(PIS56, F, $ty, Self)
                }

                pub fn pi_square() -> Self {
                    pi!(PIS56_SQUARE, F, $ty, Self)
                }

                pub fn pi_double() -> Self {
                    pi!(PIS56_DOUBLE, F, $ty, Self)
                }

                pub fn pi_half() -> Self {
                    pi!(PIS56_HALF, F, $ty, Self)
                }

                pub fn pi_quad() -> Self {
                    pi!(PIS56_QUAD, F, $ty, Self)
                }

                /// Load raw value without shifting
                pub fn load(v: $ty) -> Self {
                    Self {
                        b: v,
                        _m: PhantomData
                    }
                }

                /// `Add` values to the raw storage whith shifting
                pub fn add_raw(&mut self, v: $ty) {
                    self.b += v
                }

                /// `Mul` values to the raw storage whith shifting
                pub fn mul_raw(&mut self, v: $ty) {
                    self.b = (self.b * v) >> F::to_u8();
                }

                /// `Sub` values to the raw storage whith shifting
                pub fn sub_raw(&mut self, v: $ty) {
                    self.b -= v
                }

                /// `Div` values to the raw storage whith shifting
                pub fn div_raw(&mut self, v: $ty) {
                    self.b = (((self.b as i64) << F::to_i8()) / v as i64) as $ty
                }

                /// Convert to `T`, which will remove the fraction part.
                pub fn int(&self) -> $ty {
                    self.b >> F::to_u8()
                }

                pub fn abs(&self) -> Self {
                    Self {
                        b: self.b.abs(),
                        _m: PhantomData
                    }
                }

                pub fn is_zero(&self) -> bool {
                    self.b == 0
                }

                pub fn is_one(&self) -> bool {
                    let v = 1 as $ty << F::to_u8();
                    self.b == v
                }

                pub fn squared(self) -> Self {
                    self * self
                }

                pub fn zero() -> Self {
                    Self::load(0)
                }

                pub fn one() -> Self {
                    let v = 1 as $ty << F::to_u8();
                    Self::load(v)
                }

                pub fn eps() -> f32 {
                    1f32 / ((1u32 << F::to_u8()) as f32)
                }

                pub fn get_eps(&self) -> f32 {
                    1f32 / ((1u32 << F::to_u8()) as f32)
                }

                pub fn signum(&self) -> $ty {
                    self.b.signum()
                }

                pub fn is_positive(&self) -> bool {
                    self.b.is_positive()
                }

                pub fn is_negative(&self) -> bool {
                    self.b.is_negative()
                }

                pub fn frac(&self) -> f32 {
                    let shift = F::to_u8();
                    let i = self.b as i64;
                    (i.signum() * (((1 << shift) - 1) | i)) as f32 / (1u32 << F::to_u8()) as f32
                }

                /// Create new FPN from a float value
                pub fn new<T: Into<f64>>(v: T) -> Self {
                    Self {
                        b: (v.into() as f64 * ((1u32 << F::to_u8()) as f64)) as $ty,
                        _m: PhantomData
                    }
                }

                pub fn raw(&self) -> $ty {
                    self.b
                }

                /// Convert from `T`
                pub fn with(v: $ty) -> Self {
                    Self {
                        b: v << F::to_u8(),
                        _m: PhantomData,
                    }
                }

                pub fn to_f32(&self) -> f32 {
                    self.b as f32 / (1u32 << F::to_u8()) as f32
                }

                pub fn from_f32(v: f32) -> Self {
                    Self {
                        b: (v * ((1u32 << F::to_u8()) as f32)) as $ty,
                        _m: PhantomData,
                    }
                }

                pub fn to_f64(&self) -> f64 {
                    self.b as f64 / (1u32 << F::to_u8()) as f64
                }

                pub fn from_f64(v: f64) -> Self {
                    Self {
                        b: (v * ((1u32 << F::to_u8()) as f64)) as $ty,
                        _m: PhantomData,
                    }
                }

                pub fn to_i64(&self) -> i64 {
                    (self.b as i64) >> F::to_u8()
                }

                pub fn from_i64(v: i64) -> Self {
                    Self {
                        b: (v << F::to_u8()) as $ty,
                        _m: PhantomData,
                    }
                }

                pub fn pow(&self, v: usize) -> Self {
                    let mut p = self.b as i64;
                    let s = F::to_u8();
                    for _ in 0..v {
                        p = (p * p) >> s;
                    }
                    Self {
                        b: p as $ty,
                        _m: PhantomData
                    }
                }

                pub fn pow_assign(&mut self, v: usize) {
                    let mut p = self.b as i64;
                    let s = F::to_u8();
                    for _ in 0..v {
                        p = (p * p) >> s;
                    }
                    self.b = p as $ty;
                }

            }

            impl<F> From<FPN<$ty, F>> for f32 where F: Unsigned {
                fn from(fpn: FPN<$ty, F>) -> f32 {
                    (fpn.b as i64) as f32 / (1u32 << F::to_u8()) as f32
                }
            }

            impl<F> From<FPN<$ty, F>> for f64 where F: Unsigned {
                fn from(fpn: FPN<$ty, F>) -> f64 {
                    (fpn.b as i64) as f64 / (1u64 << F::to_u8()) as f64
                }
            }

            impl<F> Debug for FPN<$ty, F> {
                fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                    f.debug_struct("FPN").field("b", &self.b).finish()
                }
            }

            impl<F> Display for FPN<$ty, F> where F: Unsigned {
                fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                    let v: f64 = (*self).into();
                    Display::fmt(&v, f)
                }
            }

            impl<F> Neg for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        b: self.b.neg(),
                        _m: PhantomData,
                    }
                }
            }

            impl<F> Add for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Self) -> Self {
                    Self {
                        b: self.b + v.b,
                        _m: PhantomData
                    }
                }
            }

            impl<F> AddAssign for FPN<$ty, F> where F: Unsigned {
                fn add_assign(&mut self, v: Self) {
                    self.b += v.b
                }
            }

            impl<F> Sub for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Self) -> Self {
                    Self {
                        b: self.b - v.b,
                        _m: PhantomData
                    }
                }
            }

            impl<F> SubAssign for FPN<$ty, F> where F: Unsigned {
                fn sub_assign(&mut self, v: Self) {
                    self.b -= v.b
                }
            }

            impl<F> Mul for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: Self) -> Self {
                    Self {
                        b: (self.b * v.b) >> F::to_u8(),
                        _m: PhantomData
                    }
                }
            }

            impl<F> MulAssign for FPN<$ty, F> where F: Unsigned {
                fn mul_assign(&mut self, v: Self) {
                    self.b = (self.b * v.b) >> F::to_u8();
                }
            }


            impl<F> Div for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: Self) -> Self {
                    // TODO: Add round logic
                    // let round = if self.b % v.b > v.b / 2 { 1 } else { 0 };
                    Self {
                        b: (((self.b as i64) << F::to_i8()) / v.b as i64) as $ty,
                        _m: PhantomData
                    }
                }
            }

            impl<F> DivAssign for FPN<$ty, F> where F: Unsigned {
                fn div_assign(&mut self, v: Self) {
                    self.b = (((self.b as i64) << F::to_i8()) / v.b as i64) as $ty
                }
            }

            impl<F> Add<$ty> for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: $ty) -> Self {
                    Self {
                        b: self.b + (v << F::to_u8()),
                        _m: PhantomData
                    }
                }
            }

            impl<F> AddAssign<$ty> for FPN<$ty, F> where F: Unsigned {
                fn add_assign(&mut self, v: $ty) {
                    self.b += v << F::to_u8()
                }
            }

            impl<F> Sub<$ty> for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: $ty) -> Self {
                    Self {
                        b: self.b - (v << F::to_u8()),
                        _m: PhantomData
                    }
                }
            }

            impl<F> SubAssign<$ty> for FPN<$ty, F> where F: Unsigned {
                fn sub_assign(&mut self, v: $ty) {
                    self.b -= v << F::to_u8()
                }
            }

            impl<F> Mul<$ty> for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: $ty) -> Self {
                    Self {
                        b: self.b * v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> MulAssign<$ty> for FPN<$ty, F> where F: Unsigned {
                fn mul_assign(&mut self, v: $ty) {
                    self.b *= v
                }
            }


            impl<F> Div<$ty> for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: $ty) -> Self {
                    // TODO: Add round logic
                    // let round = if self.b % v.b > v.b / 2 { 1 } else { 0 };
                    Self {
                        b: self.b / v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> DivAssign<$ty> for FPN<$ty, F> where F: Unsigned {
                fn div_assign(&mut self, v: $ty) {
                    self.b /= v
                }
            }

            impl<F> Shr<u8> for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn shr(self, v: u8) -> Self {
                    Self {
                        b: self.b >> v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> ShrAssign<u8> for FPN<$ty, F> where F: Unsigned {
                fn shr_assign(&mut self, v: u8) {
                    self.b >>= v
                }
            }

            impl<F> Shl<u8> for FPN<$ty, F> where F: Unsigned {
                type Output = Self;
                fn shl(self, v: u8) -> Self {
                    Self {
                        b: self.b << v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> ShlAssign<u8> for FPN<$ty, F> where F: Unsigned {
                fn shl_assign(&mut self, v: u8) {
                    self.b <<= v
                }
            }

        )+
    }
}

impl_ops!(i8, i16, i32, i64);

macro_rules! impl_to {
    ($tf: ty; $($to: ty),+) => {
        $(
            impl<FF, FT> To<FPN<$to, FT>> for FPN<$tf, FF> where FF: Unsigned, FT: Unsigned {
                fn to(&self) -> FPN<$to, FT> {
                    let f = FF::to_u8();
                    let t = FT::to_u8();
                    if f == t {
                        FPN::<$to, FT>::load(self.b as $to)
                    } else if f < t {
                        FPN::<$to, FT>::load((self.b as $to) << (t - f))
                    } else {
                        FPN::<$to, FT>::load((self.b >> (f - t)) as $to)
                    }
                }
            }
        )+
    }
}
impl_to!(i8; i8, i16, i32, i64);
impl_to!(i16; i8, i16, i32, i64);
impl_to!(i32; i8, i16, i32, i64);
impl_to!(i64; i8, i16, i32, i64);

/*
macro_rules! impl_ops_const {
    ($i: ty, $($f: ty, $n: literal),+) => {
        $(
            impl FPN<$i, $f> {
                pub const fn with_const(v: f64) -> Self {
                    Self {
                        b: (v * ((1u32 << $n) as f64)) as $i,
                        _m: PhantomData,
                    }
                }
            }
        )+
    }
}

impl_ops_const!(i8, U1, 1, U2, 2, U3, 3, U4, 4, U5, 5, U6, 6, U7, 7, U8, 8);
*/

#[macro_export]
macro_rules! eq {
    ($a: expr, $b: expr) => {
        eq!($a, $b, $a.get_eps());
    };
    ($a: expr, $b: expr, $eps: expr) => {
        eq_with_eps!($a, $b, $eps);
    }
}

#[macro_export]
macro_rules! eq_with_eps {
    ($a: expr, $b: expr, $c: expr) => {
        assert!($a.to_f32() >= $b - $c);
        assert!($a.to_f32() <= $b + $c);
    }
}

#[cfg(test)]
mod tests {
    use crate::base::{ FPN, To };
    use typenum::{ U12, U8 };
    type F64 = FPN<i64, U12>;
    const A: f32 = 3.141592612345f32;
    const B: f32 = 9.141592612345f32;

    #[test]
    fn test_general() {
        let eps: f32 = F64::eps();
        let mut i = F64::new(A);
        assert_eq!(i.get_eps(), 1f32 / ((1u32 << 12) as f32));
        eq!(i, A);
        eq!(-i, -A);
        eq!(F64::from_f32(B), B);
        eq!(i + i, A + A, eps * 2f32);
        {
            let v = F64::from_f32(-B);
            eq!(v, -B);
            assert!(v.is_negative());
            assert!(!v.is_positive());
            eq!(v.abs(), B);
            let copy = i;
            i += v;
            assert_ne!(i, copy);
            assert!(i != copy);
        }
        assert_eq!(To::<FPN<i32, U8>>::to(&F64::new(A)), FPN::<i32, U8>::new(A));

    }

    #[test]
    fn test_ops() {
        let eps: f32 = F64::eps();
        eq!(F64::new(A) / F64::from_f32(B), A / B);
        eq!(F64::new(A) * F64::from_f32(B), A * B, eps * (A + B + 1f32));
        eq!(F64::new(A) - F64::from_f32(B), A - B);
        eq!(F64::new(A) + F64::from_f32(B), A + B, eps * 2f32);
        {
            let mut v = F64::new(A);
            let copy = v;
            let delta = (B * ((1u32 << 12) as f32)) as i64;

            v.add_raw(delta);
            eq!(v, A + B, eps * 2f32);

            v.sub_raw(delta);
            assert_eq!(v, copy);

            v.mul_raw(delta);
            eq!(v, A * B, eps * (A + B + 1f32));

            v.div_raw(delta);
            eq!(v, copy.to_f32());

            eq!(copy >> 2, A / 4f32);
            eq!(copy << 2, A * 4f32, eps * 4f32);

            let mut copy2 = copy;
            let mut copy3 = copy;
            copy2 >>= 2;
            copy3 <<= 2;
            assert_eq!(copy >> 2, copy2);
            assert_eq!(copy << 2, copy3);
        }
    }
}

/// PI << 56, (PI ** 2) << 56, (PI * 2) << 56, etc
const PIS56: i64 = 226375608064910080i64;
const PIS56_SQUARE: i64 = 711179947248643800i64;
const PIS56_DOUBLE: i64 = 452751216129820160i64;
const PIS56_HALF: i64 = 113187804032455040i64;
const PIS56_QUAD: i64 = 56593902016227520i64;
