#![feature(generic_const_exprs)]
#![allow(unused,non_snake_case)]

use anyhow::Result;
use fastcmp::Compare;
use g2p::GaloisField;
use ndarray::Array1;
use num::{Float, FromPrimitive, Integer, NumCast, Signed, ToPrimitive};
use numpy::ndarray::{ArrayView1, ArrayView2};
use ordered_float::NotNan;
use std::{
    cell::RefCell,
    cmp::min,
    collections::HashMap,
    convert::{TryFrom, TryInto},
    env::VarError,
    marker::PhantomData,
    mem::{self, transmute, MaybeUninit},
    ops::{Add, AddAssign, Range, RangeInclusive},
};
use log::debug;

macro_rules! debug_unwrap {
    ($what:expr) => {{
        #[cfg(debug_assertions)]
        {
            $what.unwrap()
        }
        #[cfg(not(debug_assertions))]
        unsafe {
            $what.unwrap_unchecked()
        }
    }};
}

/// A variable node
#[derive(Debug, Clone)]
struct VariableNode<const B: usize> {
    /// options to deal with irregular codes
    check_idx: Vec<Option<Key1D>>,
    /// The a-priory channel value
    channel: Option<QaryLlrs<B>>,
}

impl<const B: usize> Default for VariableNode<B> {
    fn default() -> Self {
        Self {
            check_idx: Vec::new(),
            channel: Default::default(),
        }
    }
}

impl<const B: usize> VariableNode<B> {
    fn new(DV: usize) -> Self {
        Self {
            check_idx: vec![None; DV],
            channel: Default::default(),
        }
    }

    fn checks(&self, var_idx: Key1D) -> impl Iterator<Item = Key2D> + '_ {
        self.check_idx
            .iter()
            .flatten()
            .map(move |check_idx| (*check_idx, var_idx).into())
    }
}

/// A check node
#[derive(Debug, Clone)]
struct CheckNode {
    /// options to deal with initialization and irregular codes
    variable_idx: Vec<Option<Key1D>>,
}

impl Default for CheckNode {
    fn default() -> Self {
        Self {
            variable_idx: Vec::new(),
        }
    }
}

impl CheckNode {
    fn new(DC: usize) -> Self {
        Self {
            variable_idx: vec![None; DC],
        }
    }

    fn variables(&self, check_idx: Key1D) -> impl Iterator<Item = Key2D> + '_ {
        self.variable_idx
            .iter()
            .flatten()
            .map(move |var_idx| (check_idx, *var_idx).into())
    }
}

pub type FloatType = f32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QaryLlrs<const B: usize>([FloatType; B]);

// Returns the arguments ordered by value
fn min_max<I: PartialOrd>(in1: I, in2: I) -> (I, I) {
    if in1 < in2 {
        (in1, in2)
    } else {
        (in2, in1)
    }
}

impl<const Q: usize> QaryLlrs<Q> {
    // q-ary'ily Add self with term
    fn qary_add(&self, term: Self) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] + term.0[q];
        }
        ret
    }

    // q-ary'ily Subtract self with subtrahend
    fn qary_sub(&self, subtrahend: Self) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] - subtrahend.0[q];
        }
        ret
    }

    fn qary_sub_arg(&self, arg_min: usize) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] - self.0[arg_min];
        }
        ret
    }

    // assume hij is 1 or -1, after multiplication array is the same (hij==1) or reversed
    fn mult_in_gf(&self, hij: i8) -> Self {
        let mut ret = Self(self.0);
        if hij < 0 {
            for q in 0..Q {
                ret.0[q] = self.0[Q - q - 1];
            }
        }
        ret
    }

    // q-ary'ily Add self with term multiplied by hij
    fn qary_add_with_mult_in_gf(&self, term: Self, hij: i8) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        if hij > 0 {
            for q in 0..Q {
                ret.0[q] = self.0[q] + term.0[q];
            }
        } else {
            for q in 0..Q {
                ret.0[q] = self.0[q] + term.0[Q - q - 1];
            }
        }
        ret
    }

    // q-ary'ily Subtract self with term multiplied by hij
    fn qary_sub_with_mult_in_gf(&self, subtrahend: Self, hij: i8) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        if hij > 0 {
            for q in 0..Q {
                ret.0[q] = self.0[q] - subtrahend.0[q];
            }
        } else {
            for q in 0..Q {
                ret.0[q] = self.0[q] - subtrahend.0[Q - q - 1];
            }
        }
        ret
    }
}

#[derive(Debug, Default, Clone)]
struct Edge<const Q: usize> {
    v2c: Option<QaryLlrs<Q>>,
    c2v: Option<QaryLlrs<Q>>,
}

type Key1D = u16;

#[allow(clippy::derive_hash_xor_eq)]
#[derive(Debug, Eq, Hash, Clone, Copy)]
struct Key2D {
    row: Key1D,
    col: Key1D,
}

impl Key2D {
    fn row(&self) -> usize {
        self.row as usize
    }
    fn col(&self) -> usize {
        self.col as usize
    }
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

impl PartialEq for Key2D {
    fn eq(&self, other: &Self) -> bool {
        let self_bytes = unsafe { any_as_u8_slice(self) };
        let other_bytes = unsafe { any_as_u8_slice(other) };
        self_bytes.feq(other_bytes)
    }
}

impl From<(usize, usize)> for Key2D {
    fn from(from: (usize, usize)) -> Self {
        unsafe {
            Self {
                row: debug_unwrap!(from.0.try_into()),
                col: debug_unwrap!(from.1.try_into()),
            }
        }
    }
}

impl From<(Key1D, Key1D)> for Key2D {
    fn from(from: (Key1D, Key1D)) -> Self {
        Self {
            row: from.0,
            col: from.1,
        }
    }
}

struct SimpleDValueIterator<BType> {
    b: BType,
    num_ignore: usize,
    SW: usize,
    d_values: Option<Vec<BType>>,
}

impl<BType> SimpleDValueIterator<BType>
where
    BType: Copy + Integer + Signed + AddAssign + NumCast,
{
    fn new(b: BType, num_ignore: usize, SW: usize) -> SimpleDValueIterator<BType> {
        assert!(num_ignore <= SW, "num_ignore must be less than or equal to SW");
        Self {
            b,
            num_ignore,
            SW,
            d_values: None,
        }
    }

    fn increment_d_values(&mut self) -> bool {
        if let Some(ref mut d_values) = self.d_values {
            for i in 0..(self.SW - self.num_ignore) {
                if d_values[i] < self.b {
                    d_values[i] += BType::from(1_usize).unwrap();
                    return true;
                }
                d_values[i] = -self.b;
            }
            false
        } else {
            let mut initial_values = vec![-self.b; self.SW];
            for i in (self.SW - self.num_ignore)..self.SW {
                initial_values[i] = BType::zero();
            }
            self.d_values = Some(initial_values);
            true
        }
    }

    pub fn next(&mut self) -> Option<&[BType]> {
        if !self.increment_d_values() {
            return None;
        }

        match &self.d_values {
            Some(ref d_values) => Some(&d_values[0..self.SW]),
            None => None,
        }
    }
}

type Container2D<T> = rustc_hash::FxHashMap<Key2D, T>;

type ParCheckType = i8;

/// Decoder that implements min_sum algorithm.
///
/// The generic arguments are:
/// DC: Maximum check node degree (num variables, per check)
/// B: first N-R variables from the range [-B, ..., 0, ..., B]
/// BSIZE: size of range [-B, ..., 0, ..., B], i.e. 2*B+1
///
/// Beside that, decoder is built from parity check matrix of size N \times R
/// N: Number of variable nodes
/// R: Number of check nodes
/// DV: Maximum variable node degree (num checks, per variable)
/// JOINTSIZE: size of range of variables that are defined as joint distribution
pub struct DecoderJointDistribution<
    const B: usize,
    const BSIZE: usize,
    const DC: usize,
    const JOINTSIZE: usize,
    BType: Integer + Signed,
> {
    DV: usize,
    N: usize,
    R: usize,
    /// Parity check matrix
    parity_check: Container2D<ParCheckType>,
    /// Messages between B-variables and check nodes
    edges: Container2D<Edge<BSIZE>>,
    /// Messages between joint variables and check nodes
    edgesjoint: Container2D<Edge<JOINTSIZE>>,
    /// List of B-Variable nodes
    vn: Vec<VariableNode<BSIZE>>,
    /// List of joint variable nodes
    vnjoint: Vec<VariableNode<JOINTSIZE>>,
    /// List of Check nodes, each node contain DC-1 B-variables and 1 joint variable
    cn: Vec<CheckNode>,
    /// Number of iterations to perform in the decoder
    max_iter: u32,
    brange: RangeInclusive<BType>,
}

fn insert_first_none<T>(array: &mut Vec<Option<T>>, value: T) {
    for el in array {
        if el.is_none() {
            el.replace(value);
            return;
        }
    }
    panic!("Reached the end of the array, no more space left!");
}

impl<
        const B: usize,
        const BSIZE: usize,
        const DC: usize,
        const JOINTSIZE: usize,
        BType,
    > DecoderJointDistribution<B, BSIZE, DC, JOINTSIZE, BType>
where
    BType: Integer + Signed + NumCast + AddAssign + Copy + FromPrimitive + TryInto<usize> + Default + std::fmt::Debug,
{
    pub const B: isize = B as isize;
    pub const BSIZE: usize = BSIZE;

    pub fn new(parity_check: Vec<Vec<ParCheckType>>, DV: usize, max_iter: u32) -> Self {
        let R = parity_check.len();
        let N = parity_check[0].len();
        let BVARS = N - R;

        let mut vn: Vec<VariableNode<BSIZE>> = (0..BVARS)
            .map(|_| VariableNode::new(DV))
            .collect();

        let mut vnjoint: Vec<VariableNode<JOINTSIZE>> = (0..R)
            .map(|_| VariableNode::new(1))
            .collect();

        let mut cn: Vec<CheckNode> = (0..R)
            .map(|_| CheckNode::new(DC))
            .collect();
        let vn_ref = RefCell::new(vn);
        let vnjoint_ref = RefCell::new(vnjoint);
        let cn_ref = RefCell::new(cn);
        let edges = RefCell::new(Container2D::default());
        let edgesjoint = RefCell::new(Container2D::default());

        let parity_check: Container2D<ParCheckType> = parity_check
            .iter()
            .enumerate()
            .flat_map(|(row_num, row)| {
                let vn = &vn_ref;
                let vnjoint = &vnjoint_ref;
                let cn = &cn_ref;
                let edges = &edges;
                let edgesjoint = &edgesjoint;
                row.iter()
                    .enumerate()
                    // Filter out zeroes in the parity check
                    .filter(|(col_num, &h)| h != 0)
                    // Handle the non-zeroes
                    .map(move |(col_num, &h)| {
                        let ij: Key2D = (row_num, col_num).into();

                        if col_num >= BVARS {
                            // This creates an empty edge (no message in either direction)
                            edgesjoint.borrow_mut().insert(ij, Default::default());
                            // add the check index to the variable
                            insert_first_none(
                                &mut vnjoint.borrow_mut()[ij.col() - BVARS].check_idx,
                                ij.row,
                            );
                        } else {
                            edges.borrow_mut().insert(ij, Default::default());
                            insert_first_none(
                                &mut vn.borrow_mut()[ij.col()].check_idx,
                                ij.row,
                            );
                        }

                        // add the variable index to the check
                        insert_first_none(
                            &mut cn.borrow_mut()[ij.row()].variable_idx,
                            ij.col,
                        );

                        (ij, h)
                    })
            })
            .collect();

        let mut vn = vn_ref.into_inner();
        let mut vnjoint = vnjoint_ref.into_inner();
        let cn = cn_ref.into_inner();
        let edges = edges.into_inner();
        let edgesjoint = edgesjoint.into_inner();

        Self {
            DV,
            N,
            R,
            parity_check,
            edges,
            edgesjoint,
            vn,
            vnjoint,
            cn,
            max_iter,
            brange: ((BType::from(-Self::B).unwrap())..=(BType::from(Self::B).unwrap())),
        }
    }

    fn initialize_channel_values<const SIZE: usize>(
        vn: &mut Vec<VariableNode<SIZE>>,
        channel_llr: &[QaryLlrs<SIZE>],
        edges: &mut Container2D<Edge<SIZE>>,
        start_var_idx: u16,
    ) {
        for (var_idx, (v, llr)) in (start_var_idx..).zip(vn.iter_mut().zip(channel_llr)) {
            v.channel = Some(*llr);
            for key in v.checks(var_idx) {
                debug_unwrap!(edges.get_mut(&key)).v2c.insert(*llr);
            }
        }
    }

    /// Formats the sum of two numbers as string.
    ///
    /// # Safety
    ///
    /// This function is safe to use if it does not panic in debug builds!
    pub fn min_sum(&self, channel_llr: Vec<QaryLlrs<BSIZE>>, channel_llr_joint: Vec<QaryLlrs<JOINTSIZE>>) -> Result<Vec<BType>> {
        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut vnjoint = self.vnjoint.clone();
        let mut edges = self.edges.clone();
        let mut edgesjoint = self.edgesjoint.clone();
        let mut hard_decision = vec![BType::zero(); self.N - self.R];

        let BVARS = self.N - self.R;
        let SW = DC - 1;

        // 0. Initialize the channel values
        Self::initialize_channel_values(
            &mut vn,
            &channel_llr,
            &mut edges,
            0,
        );
        Self::initialize_channel_values(
            &mut vnjoint,
            &channel_llr_joint,
            &mut edgesjoint,
            BVARS as u16,
        );

        let mut it = 0;
        'decoding: loop {
            it += 1;
            // 1. Parity check: Compute the syndrome based on the hard_decision
            // noop, we use only num iterations instead
            // 2. check num iterations
            // noop, we do it at the end of the loop instead
            // 3. Check node update (min)
            'check_update: for (check_idx, check) in (0..).zip(&*self.cn) {
                // Check nodes in cn are a list of values followed by some amount (potentially 0) of Nones
                // since the code is not generally regular. We assume check matrix is built as H||I,
                // therefore, for all check nodes the last non-None value corresponds to I, i.e. check value 
                let num_nones = check.variable_idx.iter().rev().take_while(|&&x| x.is_none()).count();
                let num_variable_nodes = SW - num_nones;

                let mut check_iter = check.variables(check_idx);
                let alpha_i: Vec<&QaryLlrs<BSIZE>> = check_iter
                    .by_ref()
                    .take(num_variable_nodes)
                    .map(|key| debug_unwrap!(&edges[&key].v2c.as_ref()))
                    .collect();
                let alpha_ij_joint: &QaryLlrs<JOINTSIZE> = check_iter
                    .map(|key| debug_unwrap!(&edgesjoint[&key].v2c.as_ref()))
                    .next()
                    .unwrap();

                let mut beta_i = vec![QaryLlrs::<BSIZE>([FloatType::INFINITY; BSIZE]); SW];
                let mut beta_ij_joint = QaryLlrs::<JOINTSIZE>([FloatType::INFINITY; JOINTSIZE]);
                
                // TODO: actually start supporting less than SW variables
                let mut d_values_iter = SimpleDValueIterator::<BType>::new(BType::from(B).unwrap(), num_nones, SW);
                while let Some(d_values) = d_values_iter.next() {
                    let mut sum_of_alpha: FloatType = alpha_i
                        .iter()
                        .zip(d_values)
                        .map(|(alpha_ij, d)| {
                            alpha_ij.0[Self::b2i::<B>(*d)]
                        })
                        .sum();
                    let d_values_index = Self::index_of_joint(d_values);
                    sum_of_alpha += alpha_ij_joint.0[d_values_index];

                    for ((beta_ij, d), alpha_ij) in beta_i.iter_mut().zip(d_values).zip(&alpha_i) {
                        beta_ij.0[Self::b2i::<B>(*d)] = beta_ij.0[Self::b2i::<B>(*d)]
                            .min(sum_of_alpha - alpha_ij.0[Self::b2i::<B>(*d)]);
                    }
                    beta_ij_joint.0[d_values_index] = beta_ij_joint.0[d_values_index]
                        .min(sum_of_alpha - alpha_ij_joint.0[d_values_index]);
                }

                let mut check_iter = check.variables(check_idx);
                for (key, beta_ij) in check_iter.by_ref().take(num_variable_nodes).zip(beta_i) {
                    debug_unwrap!(edges.get_mut(&key)).c2v.replace(beta_ij);
                }
                debug_unwrap!(edgesjoint.get_mut(&check_iter.next().unwrap())).c2v.replace(beta_ij_joint);
            }

            // Variable node update (sum)
            for (var_idx, var) in (0..).zip(&*vn) {
                // Collect connected checks

                // 4.1 primitive messages. Full summation
                let mut sum: QaryLlrs<BSIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edges.get(&key)).c2v);
                    sum = sum.qary_add(incoming);
                }
                for key in var.checks(var_idx) {
                    // 4.2 primitive outgoing messages, subtract self for individual message
                    let edge = debug_unwrap!(edges.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = sum.qary_sub(incoming);
                    // 5. Message normalization
                    let arg_min = Self::arg_min::<BSIZE>(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                if it >= self.max_iter {
                    // 6. Tentative decision
                    hard_decision[var_idx as usize] = Self::i2b::<B>(Self::arg_min::<BSIZE>(sum));
                }
            }
            for (var_idx, var) in ((BVARS as u16)..).zip(&*vnjoint) {
                let mut sum: QaryLlrs<JOINTSIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edgesjoint.get(&key)).c2v);
                    sum = sum.qary_add(incoming);
                }
                for key in var.checks(var_idx) {
                    let edge = debug_unwrap!(edgesjoint.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = sum.qary_sub(incoming);
                    let arg_min = Self::arg_min::<JOINTSIZE>(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                // if it >= self.max_iter {
                //     hard_decision[var_idx as usize] = Self::arg_min::<JOINTSIZE>(sum);
                // }
            }

            if it >= self.max_iter {
                break 'decoding;
            }
        }

        Ok(hard_decision)
    }

    pub fn into_llr<const T: usize>(channel_output: &Vec<Vec<FloatType>>) -> Vec<QaryLlrs<T>> {
        const EPSILON: FloatType = 0.001;
        let mut llrs: Vec<QaryLlrs<T>> = vec![QaryLlrs::<T>([0.0; T]); channel_output.len()];
        for (var, msg) in channel_output.iter().zip(llrs.iter_mut()) {
            let sum: FloatType = var.iter().sum();
            let max = var
                .iter()
                .copied()
                // ignore NAN values (errors from the previous line)
                .flat_map(NotNan::new)
                .max()
                .map(NotNan::into_inner)
                .expect("No maximum probability found");
            // Ensure the probabilities sum to 1.0, taking into
            // account the problem of floating point comparisons
            assert!(sum < 1.0 + EPSILON);
            assert!(sum > 1.0 - EPSILON);
            // calculate LLR
            for (ent, dst) in var.iter().zip(msg.0.iter_mut()) {
                *dst = (max / ent).ln();
            }
        }

        llrs
    }

    fn arg_min<const T: usize>(m: QaryLlrs<T>) -> usize {
        let mut min_val = FloatType::INFINITY;
        let mut min_arg = 0;
        for (arg, val) in m.0.iter().copied().enumerate() {
            if val < min_val {
                min_val = val;
                min_arg = arg;
            }
        }
        min_arg
    }

    pub fn i2b<const T: usize>(i: usize) -> BType {
        let i: isize = i.try_into().unwrap();
        let b: isize = T.try_into().unwrap();
        let val = i - b;
        if val > b || val < -b {
            panic!("Value over-/underflow!");
        }
        BType::from(val).unwrap()
    }

    pub fn index_of_joint(d_values: &[BType]) -> usize {
        let mut index = 0_usize;
        let mut multiplier = 1_usize;

        for &d in d_values {
            index += Self::b2i::<B>(d) * multiplier;
            multiplier *= BSIZE;
        }

        index
    }


    pub fn b2i<const T: usize>(val: BType) -> usize
    where
        BType: TryInto<usize>,
    {
        let mut val: isize = val.to_isize().unwrap();

        (val + (T as isize)) as usize
    }
}

fn into_or_panic<T, U>(from: T) -> U
where
    T: TryInto<U>,
{
    match from.try_into() {
        Ok(val) => val,
        Err(_) => panic!("Failed conversion!"),
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn into_llr() {
        /* let channel_output = [[
            0.0, 0.0, 0.0, 0.0, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.02, 0.0, 0.0, 0.0,
        ]; MyTinyTestDecoder::N];
        let llr = MyTinyTestDecoder::into_llr(&channel_output);
        let expected = [QaryLlrs([
            FloatType::INFINITY,
            FloatType::INFINITY,
            FloatType::INFINITY,
            FloatType::INFINITY,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            //1.9459101490553135,
            1.945_910_1,
            FloatType::INFINITY,
            FloatType::INFINITY,
            FloatType::INFINITY,
        ]); MyTinyTestDecoder::N];
        assert_eq!(expected, llr); */
    }

    #[test]
    fn it_works() {
/*         let decoder_6_3_4_3_gf16 = MyTinyTestDecoder::new(
            [
                [true, true, true, true, false, false],
                [false, false, true, true, false, true],
                [true, false, false, true, true, false],
            ],
            10,
        );

        // Zero message with zero noise
        let mut channel_output = [[0.0; MyTinyTestDecoder::Q]; MyTinyTestDecoder::N];
        for el in &mut channel_output {
            el[MyTinyTestDecoder::b2i(0)] = 1.0;
        }

        // Introduce an error
        channel_output[1][MyTinyTestDecoder::b2i(0)] = 0.1;
        channel_output[1][MyTinyTestDecoder::b2i(7)] = 0.9;

        // Convert to LLR
        let channel_llr = MyTinyTestDecoder::into_llr(&channel_output);

        let res = decoder_6_3_4_3_gf16.min_sum(channel_llr).expect("Failed");

        let expected: [i8; 6] = [0; 6];

        assert_eq!(res, expected); */
    }
}
