//! `base` provides basic fixed point numbers and implemented common `ops` traits for FPNs with integar primitives.
//! 

use core::marker::PhantomData;
use core::cmp::{ self, Ordering };
use typenum::*;
use std::ops::*;
use std::fmt::{ self, * };

type F64 = FPN<i64, U12>;

pub trait Pri {}
pub trait PriInt {}
pub trait PriFloat {}

macro_rules! impl_pri {
    ($trait: ty, $($I: ty),+) => {
        $(
            impl $trait for $I {}
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
    ($v:expr, $F: ident, $I: ty, $Self: ident, $s: expr) => {
        {
            let s = $F::to_u8();
            let b = if s < $s {
                $v >> $s - s
            } else {
                $v << s - $s
            };
            $Self {
                b: b as $I,
                _m: PhantomData
            }
        }
    }
}

macro_rules! impl_ops {
    ($($I: ty, $T: ident),+) => {
        $(
            impl<F> FPN<$I, F> where F: Unsigned {
                pub fn min() -> Self {
                    Self {
                        b: $T::MIN,
                        _m: PhantomData
                    }
                }

                pub fn max() -> Self {
                    Self {
                        b: $T::MAX,
                        _m: PhantomData
                    }
                }

                /// `cos` is calculated with prepared list of cos and sin values for angle from 0
                /// to pi/4. The difference to value of float normally should be pretty small 
                /// around 0.0039, but
                /// it could be a bit big when curve goes steep, with this case, it's normally
                /// still below 0.05.
                pub fn cos(&self) -> Self {
                    let pi_double = F64::pi_double();
                    let pi = pi_double >> 1;
                    let pi_half = pi >> 1;
                    let pi_quad = pi_half >> 1;
                    let mut sig = 1;
                    let mut angle: F64 = self.abs().to();
                    angle = angle % pi_double;
                    if angle > pi {
                        angle = pi_double - angle;
                    }
                    if angle > pi_half {
                        sig = -1;
                        angle = pi - angle;
                    }
                    if angle > pi_quad {
                        // use sin
                        angle = pi_half - angle;
                        (F64::load(SINS[((angle.b * 64) / pi_quad.b) as usize]) * sig).to()
                    } else {
                        // use cos
                        (F64::load(COSS[((angle.b * 64) / pi_quad.b) as usize]) * sig).to()
                    }
                }

                /// `sin` is calculated with prepared list of cos and sin values for angle from 0
                /// to pi/4. The difference to value of float normally should be pretty small which is
                /// around 0.0039, but
                /// it could be a bit big when curve goes steep, with this case, it's normally
                /// still below 0.05. In implementation, `sin(angle) = cos(angle - pi/2)`.
                pub fn sin(&self) -> Self {
                    let angle: F64 = self.to();
                    (angle - F64::pi_half()).cos().to()
                }

                /// `tan` is calulated with prepare list of tan values for angles from 0 to pi/2.
                /// The difference to the value of the float is normally pretty small. However for
                /// values bigger than 1 or smaller than -1, the curve starts going steep, and
                /// the difference at this case could be bigger and non-negligible.
                pub fn tan(&self) -> Self {
                    let pi = F64::pi();
                    let pi_half = pi >> 1;
                    let pi_quad = pi_half >> 1;
                    let mut angle: F64 = self.to();
                    let mut sig = angle.signum();
                    angle = angle.abs() % pi;
                    if angle > pi_half {
                        sig = -sig;
                        angle = pi - angle;
                    }
                    (F64::load(TANS[((angle.b * 64) / pi_quad.b) as usize]) * sig).to()
                }

                pub fn pi() -> Self {
                    pi!(PIS56, F, $I, Self, 56)
                }

                pub fn pi_square() -> Self {
                    pi!(PIS56_SQUARE, F, $I, Self, 56)
                }

                pub fn pi_double() -> Self {
                    pi!(PIS56, F, $I, Self, 55)
                }

                pub fn pi_half() -> Self {
                    pi!(PIS56, F, $I, Self, 57)
                }

                pub fn pi_quad() -> Self {
                    pi!(PIS56, F, $I, Self, 58)
                }

                /// Load raw value without shifting
                pub fn load(v: $I) -> Self {
                    Self {
                        b: v,
                        _m: PhantomData
                    }
                }

                /// `Add` values to the raw storage whith shifting
                pub fn add_raw(&mut self, v: $I) {
                    self.b += v
                }

                /// `Mul` values to the raw storage whith shifting
                pub fn mul_raw(&mut self, v: $I) {
                    self.b = (self.b * v) >> F::to_u8();
                }

                /// `Sub` values to the raw storage whith shifting
                pub fn sub_raw(&mut self, v: $I) {
                    self.b -= v
                }

                /// `Div` values to the raw storage whith shifting
                pub fn div_raw(&mut self, v: $I) {
                    self.b = (((self.b as i64) << F::to_i8()) / v as i64) as $I
                }

                /// Convert to `T`, which will remove the fraction part.
                pub fn int(&self) -> $I {
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
                    let v = 1 as $I << F::to_u8();
                    self.b == v
                }

                pub fn squared(self) -> Self {
                    self * self
                }

                pub fn zero() -> Self {
                    Self::load(0)
                }

                pub fn one() -> Self {
                    let v = 1 as $I << F::to_u8();
                    Self::load(v)
                }

                pub fn eps() -> f32 {
                    1f32 / ((1u32 << F::to_u8()) as f32)
                }

                pub fn get_eps(&self) -> f32 {
                    1f32 / ((1u32 << F::to_u8()) as f32)
                }

                pub fn signum(&self) -> $I {
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
                        b: (v.into() as f64 * ((1u32 << F::to_u8()) as f64)) as $I,
                        _m: PhantomData
                    }
                }

                pub fn raw(&self) -> $I {
                    self.b
                }

                /// Convert from `T`
                pub fn with(v: $I) -> Self {
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
                        b: (v * ((1u32 << F::to_u8()) as f32)) as $I,
                        _m: PhantomData,
                    }
                }

                pub fn to_f64(&self) -> f64 {
                    self.b as f64 / (1u32 << F::to_u8()) as f64
                }

                pub fn from_f64(v: f64) -> Self {
                    Self {
                        b: (v * ((1u32 << F::to_u8()) as f64)) as $I,
                        _m: PhantomData,
                    }
                }

                pub fn to_i64(&self) -> i64 {
                    (self.b as i64) >> F::to_u8()
                }

                pub fn from_i64(v: i64) -> Self {
                    Self {
                        b: (v << F::to_u8()) as $I,
                        _m: PhantomData,
                    }
                }

                pub fn pow(&self, v: usize) -> Self {
                    let mut p = self.b as i64;
                    let s = F::to_u8();
                    for _ in 0..v - 1 {
                        p = (p * p) >> s;
                    }
                    Self {
                        b: p as $I,
                        _m: PhantomData
                    }
                }

                pub fn pow_assign(&mut self, v: usize) {
                    let mut p = self.b as i64;
                    let s = F::to_u8();
                    for _ in 0..v - 1 {
                        p = (p * p) >> s;
                    }
                    self.b = p as $I;
                }

            }

            impl<F> From<FPN<$I, F>> for f32 where F: Unsigned {
                fn from(fpn: FPN<$I, F>) -> f32 {
                    (fpn.b as i64) as f32 / (1u32 << F::to_u8()) as f32
                }
            }

            impl<F> From<FPN<$I, F>> for f64 where F: Unsigned {
                fn from(fpn: FPN<$I, F>) -> f64 {
                    (fpn.b as i64) as f64 / (1u64 << F::to_u8()) as f64
                }
            }

            impl<F> Debug for FPN<$I, F> {
                fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                    f.debug_struct("FPN").field("b", &self.b).finish()
                }
            }

            impl<F> Display for FPN<$I, F> where F: Unsigned {
                fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                    let v: f64 = (*self).into();
                    Display::fmt(&v, f)
                }
            }

            impl<F> Neg for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Self::Output {
                        b: self.b.neg(),
                        _m: PhantomData,
                    }
                }
            }

            impl<F> Add for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: Self) -> Self {
                    Self {
                        b: self.b + v.b,
                        _m: PhantomData
                    }
                }
            }

            impl<F> AddAssign for FPN<$I, F> where F: Unsigned {
                fn add_assign(&mut self, v: Self) {
                    self.b += v.b
                }
            }

            impl<F> Sub for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: Self) -> Self {
                    Self {
                        b: self.b - v.b,
                        _m: PhantomData
                    }
                }
            }

            impl<F> SubAssign for FPN<$I, F> where F: Unsigned {
                fn sub_assign(&mut self, v: Self) {
                    self.b -= v.b
                }
            }

            impl<F> Mul for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: Self) -> Self {
                    Self {
                        b: (self.b * v.b) >> F::to_u8(),
                        _m: PhantomData
                    }
                }
            }

            impl<F> MulAssign for FPN<$I, F> where F: Unsigned {
                fn mul_assign(&mut self, v: Self) {
                    self.b = (self.b * v.b) >> F::to_u8();
                }
            }


            impl<F> Div for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: Self) -> Self {
                    // TODO: Add round logic?
                    // let round = if self.b % v.b > v.b / 2 { 1 } else { 0 };
                    Self {
                        b: (((self.b as i64) << F::to_i8()) / v.b as i64) as $I,
                        _m: PhantomData
                    }
                }
            }

            impl<F> DivAssign for FPN<$I, F> where F: Unsigned {
                fn div_assign(&mut self, v: Self) {
                    self.b = (((self.b as i64) << F::to_i8()) / v.b as i64) as $I
                }
            }

            impl<F> Rem for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn rem(self, v: Self) -> Self {
                    Self {
                        b: ((self.b as i64) % v.b as i64) as $I,
                        _m: PhantomData
                    }
                }
            }

            impl<F> RemAssign for FPN<$I, F> where F: Unsigned {
                fn rem_assign(&mut self, v: Self) {
                    self.b = ((self.b as i64) % v.b as i64) as $I
                }
            }

            impl<F> Add<$I> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn add(self, v: $I) -> Self {
                    Self {
                        b: self.b + (v << F::to_u8()),
                        _m: PhantomData
                    }
                }
            }

            impl<F> AddAssign<$I> for FPN<$I, F> where F: Unsigned {
                fn add_assign(&mut self, v: $I) {
                    self.b += v << F::to_u8()
                }
            }

            impl<F> Sub<$I> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn sub(self, v: $I) -> Self {
                    Self {
                        b: self.b - (v << F::to_u8()),
                        _m: PhantomData
                    }
                }
            }

            impl<F> SubAssign<$I> for FPN<$I, F> where F: Unsigned {
                fn sub_assign(&mut self, v: $I) {
                    self.b -= v << F::to_u8()
                }
            }

            impl<F> Mul<$I> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn mul(self, v: $I) -> Self {
                    Self {
                        b: self.b * v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> MulAssign<$I> for FPN<$I, F> where F: Unsigned {
                fn mul_assign(&mut self, v: $I) {
                    self.b *= v
                }
            }


            impl<F> Div<$I> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn div(self, v: $I) -> Self {
                    // TODO: Add round logic
                    // let round = if self.b % v.b > v.b / 2 { 1 } else { 0 };
                    Self {
                        b: self.b / v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> DivAssign<$I> for FPN<$I, F> where F: Unsigned {
                fn div_assign(&mut self, v: $I) {
                    self.b /= v
                }
            }

            impl<F> Rem<$I> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn rem(self, v: $I) -> Self {
                    Self {
                        b: ((self.b as i64) % ((v as i64) << F::to_u8())) as $I,
                        _m: PhantomData
                    }
                }
            }

            impl<F> RemAssign<$I> for FPN<$I, F> where F: Unsigned {
                fn rem_assign(&mut self, v: $I) {
                    self.b = ((self.b as i64) % ((v as i64) << F::to_u8())) as $I
                }
            }

            impl<F> Shr<u8> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn shr(self, v: u8) -> Self {
                    Self {
                        b: self.b >> v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> ShrAssign<u8> for FPN<$I, F> where F: Unsigned {
                fn shr_assign(&mut self, v: u8) {
                    self.b >>= v
                }
            }

            impl<F> Shl<u8> for FPN<$I, F> where F: Unsigned {
                type Output = Self;
                fn shl(self, v: u8) -> Self {
                    Self {
                        b: self.b << v,
                        _m: PhantomData
                    }
                }
            }

            impl<F> ShlAssign<u8> for FPN<$I, F> where F: Unsigned {
                fn shl_assign(&mut self, v: u8) {
                    self.b <<= v
                }
            }

        )+
    }
}

impl_ops!(i8, i8, i16, i16, i32, i32, i64, i64);

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
    use rand::prelude::*;

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

        let n = 3993i64;
        assert_eq!(F64::with(n).pow(2).to_i64(), n.pow(2));
    }

    #[test]
    fn test_pi() {
        eq!(F64::pi_quad(), std::f32::consts::FRAC_PI_4);
        eq!(F64::pi_half(), std::f32::consts::FRAC_PI_2);
        eq!(F64::pi(), std::f32::consts::PI);
        eq!(F64::pi_double(), std::f32::consts::PI * 2f32);
        eq!(F64::pi_square(), std::f32::consts::PI * std::f32::consts::PI);
    }

    #[test]
    fn test_tri() {
        let mut rng = rand::thread_rng();
        macro_rules! test_tri {
            ($m: ident, $name: expr, $d: expr) => {
                {
                    let mut min = f32::MAX;
                    let mut max = 0f32;
                    let mut avg = 0f32;
                    let mut a;
                    for i in 0..1000 {
                        a = i as f32 * std::f32::consts::FRAC_PI_2 / 1000f32;
                        let o = F64::new(a).$m().to_f32();
                        let p = a.$m();
                        let d = (o - p).abs();
                        avg += d;
                        if d < min {
                            min = d;
                        }
                        if d > max {
                            max = d;
                            println!("BIGGER {} {} {} {}", a * 180f32 / std::f32::consts::PI, d, o, p);
                        }
                        // println!("{} : {} {}  {}", d, a, o, p);
                    }

                    println!("{} difference: min {} max {} avg {}", $name, min, max, avg / 1000f32);

                    for _ in 0..10000 {
                        let angle: f64 = rng.gen();
                        let a = F64::new(angle * 1000f64);
                        let i = a.$m().to_f32();
                        let j = a.to_f32().$m();
                        // println!("{} {} {}", i - j, i, j);
                        eq!(a.$m(), j, $d);
                        eq!(a.$m(), (a + F64::pi_double()).$m().to_f32(), $d);
                    }
                    println!("-----------------------------------------------------------");
                }
            }
        }
        test_tri!(sin, "sin", 0.05);
        test_tri!(cos, "cos", 0.05);
        test_tri!(tan, "tan", std::f32::MAX);
        // assert!(false);
    }

    #[test]
    fn test_ops() {
        let eps: f32 = F64::eps();
        eq!(F64::new(A) / F64::from_f32(B), A / B);
        eq!(F64::new(A) * F64::from_f32(B), A * B, eps * (A + B + 1f32));
        eq!(F64::new(A) - F64::from_f32(B), A - B);
        eq!(F64::new(A) + F64::from_f32(B), A + B, eps * 2f32);
        eq!(F64::new(B) % F64::from_f32(A), B % A);
        eq!(F64::new(A) % F64::from_f32(B), A % B);
        // println!("{} {}", (F64::new(B) % F64::from_f32(A)).to_f32(), B % A);
        // println!("{} {}", (F64::new(A) % F64::from_f32(B)).to_f32(), A % B);
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

/// Constants used to convert values from cartesian coordinates to polar coordinates
/// Constant tan(a) values with a from (PI/4) * 0 / 64 to (PI/4) * 128 / 64
/// Constant sin(a) and cos(a) values with a from (PI/4) * 0 / 64 to (PI/4) * 64 / 64
pub const TANS: [i64; 129] = [
    0i64, 50i64, 101i64, 151i64, 201i64, 252i64, 302i64, 353i64, 403i64, 454i64,
    505i64, 556i64, 608i64, 659i64, 711i64, 763i64, 815i64, 867i64, 920i64, 973i64,
    1026i64, 1080i64, 1134i64, 1188i64, 1243i64, 1298i64, 1353i64, 1409i64, 1466i64, 1523i64,
    1580i64, 1638i64, 1697i64, 1756i64, 1816i64, 1876i64, 1937i64, 1999i64, 2062i64, 2125i64,
    2189i64, 2254i64, 2320i64, 2387i64, 2455i64, 2524i64, 2594i64, 2665i64, 2737i64, 2810i64,
    2885i64, 2961i64, 3038i64, 3116i64, 3197i64, 3278i64, 3362i64, 3446i64, 3533i64, 3622i64,
    3712i64, 3805i64, 3900i64, 3997i64, 4096i64, 4198i64, 4302i64, 4409i64, 4519i64, 4632i64,
    4748i64, 4868i64, 4991i64, 5118i64, 5249i64, 5383i64, 5523i64, 5667i64, 5816i64, 5970i64,
    6130i64, 6296i64, 6468i64, 6647i64, 6834i64, 7028i64, 7230i64, 7442i64, 7663i64, 7895i64,
    8137i64, 8392i64, 8660i64, 8943i64, 9240i64, 9555i64, 9889i64, 10242i64, 10618i64, 11019i64,
    11448i64, 11906i64, 12399i64, 12929i64, 13503i64, 14124i64, 14801i64, 15540i64, 16352i64, 17247i64,
    18240i64, 19348i64, 20592i64, 22000i64, 23606i64, 25457i64, 27613i64, 30158i64, 33209i64, 36935i64,
    41587i64, 47564i64, 55528i64, 66671i64, 83376i64, 111207i64, 166853i64, 333755i64,
    std::i64::MAX,
];
pub const COSS: [i64; 65] = [
    4096i64, 4096i64, 4095i64, 4093i64, 4091i64, 4088i64, 4085i64, 4081i64, 4076i64, 4071i64,
    4065i64, 4059i64, 4052i64, 4044i64, 4036i64, 4027i64, 4017i64, 4007i64, 3996i64, 3985i64,
    3973i64, 3961i64, 3948i64, 3934i64, 3920i64, 3905i64, 3889i64, 3873i64, 3857i64, 3839i64,
    3822i64, 3803i64, 3784i64, 3765i64, 3745i64, 3724i64, 3703i64, 3681i64, 3659i64, 3636i64,
    3612i64, 3588i64, 3564i64, 3539i64, 3513i64, 3487i64, 3461i64, 3433i64, 3406i64, 3378i64,
    3349i64, 3320i64, 3290i64, 3260i64, 3229i64, 3198i64, 3166i64, 3134i64, 3102i64, 3068i64,
    3035i64, 3001i64, 2967i64, 2932i64, 2896i64,
];
pub const SINS: [i64; 65] = [
    0i64, 50i64, 101i64, 151i64, 201i64, 251i64, 301i64, 351i64, 401i64, 451i64,
    501i64, 551i64, 601i64, 651i64, 700i64, 750i64, 799i64, 848i64, 897i64, 946i64,
    995i64, 1044i64, 1092i64, 1141i64, 1189i64, 1237i64, 1285i64, 1332i64, 1380i64, 1427i64,
    1474i64, 1521i64, 1567i64, 1614i64, 1660i64, 1706i64, 1751i64, 1797i64, 1842i64, 1886i64,
    1931i64, 1975i64, 2019i64, 2062i64, 2106i64, 2149i64, 2191i64, 2234i64, 2276i64, 2317i64,
    2359i64, 2399i64, 2440i64, 2480i64, 2520i64, 2559i64, 2598i64, 2637i64, 2675i64, 2713i64,
    2751i64, 2788i64, 2824i64, 2861i64, 2896i64,
];
/// PI << 56, (PI ** 2) << 56, (PI * 2) << 56, etc
pub const PIS56: i64 = 226375608064910080i64;
pub const PIS56_SQUARE: i64 = 711179947248643800i64;
