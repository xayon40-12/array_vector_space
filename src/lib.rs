pub trait ArrayVectorSpace<T> {
    fn dot(self, rhs: Self) -> T;
    fn norm2(self) -> T
    where
        Self: Sized + Copy,
    {
        self.dot(self)
    }
    fn add(self, rhs: Self) -> Self;
    fn sub(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn div(self, rhs: Self) -> Self;
    fn scal_mul(self, rhs: T) -> Self;
    fn normalized(self) -> Self;
}

macro_rules! impl_vector_space {
    ($t: ty) => {
        impl ArrayVectorSpace<$t> for $t {
            fn dot(self, rhs: Self) -> $t {
                self * rhs
            }
            fn add(self, rhs: Self) -> Self {
                self + rhs
            }
            fn sub(self, rhs: Self) -> Self {
                self - rhs
            }
            fn mul(self, rhs: Self) -> Self {
                self * rhs
            }
            fn div(self, rhs: Self) -> Self {
                self / rhs
            }
            fn scal_mul(self, rhs: $t) -> Self {
                self * rhs
            }
            fn normalized(self) -> Self {
                1.0
            }
        }
        impl<const N: usize, V: ArrayVectorSpace<$t> + Copy> ArrayVectorSpace<$t> for [V; N] {
            fn dot(self, rhs: Self) -> $t {
                self.into_iter()
                    .zip(rhs.into_iter())
                    .map(|(v, w)| v.dot(w))
                    .fold(0.0, <$t>::add)
            }
            fn add(mut self, rhs: Self) -> Self {
                self.iter_mut()
                    .zip(rhs.into_iter())
                    .for_each(|(v, w)| *v = v.add(w));
                self
            }
            fn sub(mut self, rhs: Self) -> Self {
                self.iter_mut()
                    .zip(rhs.into_iter())
                    .for_each(|(v, w)| *v = v.sub(w));
                self
            }
            fn mul(mut self, rhs: Self) -> Self {
                self.iter_mut()
                    .zip(rhs.into_iter())
                    .for_each(|(v, w)| *v = v.mul(w));
                self
            }
            fn div(mut self, rhs: Self) -> Self {
                self.iter_mut()
                    .zip(rhs.into_iter())
                    .for_each(|(v, w)| *v = v.div(w));
                self
            }
            fn scal_mul(mut self, rhs: $t) -> Self {
                self.iter_mut().for_each(|v| *v = v.scal_mul(rhs));
                self
            }
            fn normalized(self) -> Self {
                let n = self.norm2().sqrt();
                self.scal_mul(n.recip())
            }
        }
    };
}

impl_vector_space! {f32}
impl_vector_space! {f64}
