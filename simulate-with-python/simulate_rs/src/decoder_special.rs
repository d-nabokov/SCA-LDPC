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
use logsumexp::{LogAddExp, LogSumExp};
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
    fn qary_add(&self, term: &Self) -> Self {
        let mut ret = Self([FloatType::INFINITY; Q]);
        for q in 0..Q {
            ret.0[q] = self.0[q] + term.0[q];
        }
        ret
    }

    // q-ary'ily Subtract self with subtrahend
    fn qary_sub(&self, subtrahend: &Self) -> Self {
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
    fn qary_add_with_mult_in_gf(&self, term: &Self, hij: i8) -> Self {
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
    fn qary_sub_with_mult_in_gf(&self, subtrahend: &Self, hij: i8) -> Self {
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

pub trait IntegerLike:
    Integer + Signed + NumCast + AddAssign + Copy + FromPrimitive + TryInto<usize> + Default + std::fmt::Debug
{}

impl<T> IntegerLike for T
where
    T: Integer + Signed + NumCast + AddAssign + Copy + FromPrimitive + TryInto<usize> + Default + std::fmt::Debug,
{}

pub trait CombinationOps<BType: IntegerLike> {
    fn index_of_comb_val(val: BType, comb_weight: usize) -> usize;
    fn index_of_comb(d_values: &[BType]) -> usize;

    fn b2i<const T: usize>(val: BType) -> usize {
        let mut val: isize = val.to_isize().unwrap();

        (val + (T as isize)) as usize
    }

    fn i2b<const T: usize>(i: usize) -> BType {
        let i: isize = i.try_into().unwrap();
        let b: isize = T.try_into().unwrap();
        let val = i - b;
        if val > b || val < -b {
            panic!("Value over-/underflow!");
        }
        BType::from(val).unwrap()
    }
}

pub struct SumCombination<
    BType: IntegerLike,
    const B: usize,
> {
    b_type: PhantomData<BType>,
}

impl<
    BType: IntegerLike,
    const B: usize,
> CombinationOps<BType> for SumCombination<BType, B>
{
    fn index_of_comb_val(val: BType, comb_weight: usize) -> usize {
        let mut val: isize = val.to_isize().unwrap();

        (val + (comb_weight * B) as isize) as usize
    }

    fn index_of_comb(d_values: &[BType]) -> usize {
        let mut d_value_comb = BType::zero();
        for d in d_values.iter() {
            d_value_comb += *d;
        }
        d_value_comb = -d_value_comb;

        Self::index_of_comb_val(d_value_comb, d_values.len())
    }
}

pub struct JointCombination<
    BType: IntegerLike,
    const B: usize,
    const BSIZE: usize,
> {
    b_type: PhantomData<BType>,
}

impl<
    BType: IntegerLike,
    const B: usize,
    const BSIZE: usize,
> CombinationOps<BType> for JointCombination<BType, B, BSIZE>
{
    fn index_of_comb_val(val: BType, comb_weight: usize) -> usize {
        unimplemented!();
    }

    fn index_of_comb(d_values: &[BType]) -> usize {
        let mut index = 0_usize;
        let mut multiplier = 1_usize;

        for &d in d_values {
            index += Self::b2i::<B>(d) * multiplier;
            multiplier *= BSIZE;
        }

        index
    }
}

pub trait ParityCheckOps {
    fn mult_by_parity<const SIZE: usize>(
        incoming: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE>;

    fn add_with_mult_by_parity<const SIZE: usize>(
        lhs: &QaryLlrs<SIZE>,
        rhs: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE>;

    fn sub_with_mult_by_parity_then_mult<const SIZE: usize>(
        lhs: &QaryLlrs<SIZE>,
        rhs: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE>;
}

pub struct TernaryParityCheckOps;

impl ParityCheckOps for TernaryParityCheckOps {
    fn mult_by_parity<const SIZE: usize>(
        incoming: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE> {
        incoming.mult_in_gf(check_val)
    }

    fn add_with_mult_by_parity<const SIZE: usize>(
        lhs: &QaryLlrs<SIZE>,
        rhs: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE> {
        lhs.qary_add_with_mult_in_gf(rhs, check_val)
    }

    fn sub_with_mult_by_parity_then_mult<const SIZE: usize>(
        lhs: &QaryLlrs<SIZE>,
        rhs: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE> {
        Self::mult_by_parity(&lhs.qary_sub_with_mult_in_gf(rhs, check_val), check_val)
    }
}

pub struct BinaryParityCheckOps;

impl ParityCheckOps for BinaryParityCheckOps {
    fn mult_by_parity<const SIZE: usize>(
        incoming: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE> {
        // since we only have 0 or 1, multiplication is always by 1, i.e. nothing happens
        *incoming
    }

    fn add_with_mult_by_parity<const SIZE: usize>(
        lhs: &QaryLlrs<SIZE>,
        rhs: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE> {
        lhs.qary_add(rhs)
    }

    fn sub_with_mult_by_parity_then_mult<const SIZE: usize>(
        lhs: &QaryLlrs<SIZE>,
        rhs: &QaryLlrs<SIZE>,
        check_val: ParCheckType,
    ) -> QaryLlrs<SIZE> {
        lhs.qary_sub(rhs)
    }
}


/// Decoder implementing belief propagation algorithm. 
///
/// The generic arguments are:
/// B: first N-R variables from the range [-B, ..., 0, ..., B]
/// BSIZE: size of range [-B, ..., 0, ..., B], i.e. 2*B+1
/// COMB_SIZE: size of range of a combination of variables
///
/// Beside that, decoder is built from parity check matrix of size N \times R
/// N: Number of variable nodes
/// R: Number of check nodes
/// DC: Maximum check node degree (num variables, per check)
/// DV: Maximum variable node degree (num checks, per variable)
pub struct DecoderSpecial<
    C: CombinationOps<BType>,
    P: ParityCheckOps,
    const B: usize,
    const BSIZE: usize,
    const COMB_SIZE: usize,
    BType: IntegerLike,
> {
    /// Parity check matrix
    parity_check: Container2D<ParCheckType>,
    /// Messages between B-variables and check nodes
    edges: Container2D<Edge<BSIZE>>,
    /// Messages between BSUM-variables and check nodes
    edges_comb: Container2D<Edge<COMB_SIZE>>,
    /// List of B-Variable nodes
    vn: Vec<VariableNode<BSIZE>>,
    /// List of BSUM-Variable nodes
    vn_comb: Vec<VariableNode<COMB_SIZE>>,
    /// List of Check nodes, each node contain DC-1 B-variables and 1 BSUM-variable
    cn: Vec<CheckNode>,
    /// Number of iterations to perform in the decoder    
    max_iter: u32,
    DV: usize,
    DC: usize,
    N: usize,
    R: usize,
    combination_ops_type: PhantomData<C>,
    parity_ops_type: PhantomData<P>,
    b_type: PhantomData<BType>,
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

// Normalize log probabilities in-place so that exp(...) sums to 1
fn normalize_log_probs(log_probs: &mut [FloatType]) {
    let lse = log_probs.iter().ln_sum_exp();
    for lp in log_probs.iter_mut() {
        *lp -= lse;
    }
}

fn check_all_finite_assert<const Q: usize>(
    llrs: &[QaryLlrs<Q>],
    name: &str,
) {
    for (i, qllr) in llrs.iter().enumerate() {
        for (j, &val) in qllr.0.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Found non-finite value in {} at index ({}, {}): {}",
                name, i, j, val
            );
        }
    }
}

impl<
    C: CombinationOps<BType>,
    P: ParityCheckOps,
    const B: usize,
    const BSIZE: usize,
    const COMB_SIZE: usize,
    BType: IntegerLike,
> DecoderSpecial<C, P, B, BSIZE, COMB_SIZE, BType> {
    pub const B: isize = B as isize;
    pub const BSIZE: usize = BSIZE;
    pub const COMB_SIZE: usize = COMB_SIZE;

    pub fn new(
        parity_check: Vec<Vec<ParCheckType>>,
        DV: usize,
        DC: usize,
        max_iter: u32,
    ) -> Self {
        let R = parity_check.len();
        let N = parity_check[0].len();
        let BVARS = N - R;

        let mut vn: Vec<VariableNode<BSIZE>> = (0..BVARS)
            .map(|_| VariableNode::new(DV))
            .collect();

        let mut vn_comb: Vec<VariableNode<COMB_SIZE>> = (0..R)
            .map(|_| VariableNode::new(1))
            .collect();

        let mut cn: Vec<CheckNode> = (0..R)
            .map(|_| CheckNode::new(DC))
            .collect();
        let vn_ref = RefCell::new(vn);
        let vn_comb_ref = RefCell::new(vn_comb);
        let cn_ref = RefCell::new(cn);
        let edges = RefCell::new(Container2D::default());
        let edges_comb = RefCell::new(Container2D::default());

        let parity_check: Container2D<ParCheckType> = parity_check
            .iter()
            .enumerate()
            .flat_map(|(row_num, row)| {
                let vn = &vn_ref;
                let vn_comb = &vn_comb_ref;
                let cn = &cn_ref;
                let edges = &edges;
                let edges_comb = &edges_comb;
                row.iter()
                    .enumerate()
                    // Filter out zeroes in the parity check
                    .filter(|(col_num, &h)| h != 0)
                    // Handle the non-zeroes
                    .map(move |(col_num, &h)| {
                        let ij: Key2D = (row_num, col_num).into();

                        if col_num >= BVARS {
                            // This creates an empty edge (no message in either direction)
                            edges_comb.borrow_mut().insert(ij, Default::default());
                            // add the check index to the variable
                            insert_first_none(
                                &mut vn_comb.borrow_mut()[ij.col() - BVARS].check_idx,
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
        let mut vn_comb = vn_comb_ref.into_inner();
        let cn = cn_ref.into_inner();
        let edges = edges.into_inner();
        let edges_comb = edges_comb.into_inner();

        Self {
            parity_check,
            vn,
            vn_comb,
            cn,
            edges,
            edges_comb,
            max_iter,
            DV,
            DC,
            N,
            R,
            combination_ops_type: PhantomData,
            parity_ops_type: PhantomData,
            b_type: PhantomData,
        }
    }

    fn initialize_channel_values<const SIZE: usize>(
        &self,
        vn: &mut Vec<VariableNode<SIZE>>,
        channel_llr: &[QaryLlrs<SIZE>],
        edges: &mut Container2D<Edge<SIZE>>,
        start_var_idx: u16,
    ) {
        for (var_idx, (v, llr)) in (start_var_idx..).zip(vn.iter_mut().zip(channel_llr)) {
            v.channel = Some(*llr);
            for key in v.checks(var_idx) {
                debug_unwrap!(edges.get_mut(&key)).v2c.insert(P::mult_by_parity(llr, self.parity_check[&key]));
            }
        }
    }

    pub fn min_sum(
        &self, 
        channel_llr: Vec<QaryLlrs<BSIZE>>, 
        channel_llr_comb: Vec<QaryLlrs<COMB_SIZE>>
    ) -> Result<Vec<BType>> {
        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut vn_comb = self.vn_comb.clone();
        let mut edges = self.edges.clone();
        let mut edges_comb = self.edges_comb.clone();
        
        let BVARS = self.N - self.R;
        let SW = self.DC - 1;

        let mut hard_decision = vec![BType::zero(); BVARS];

        // 0. Initialize the channel values
        self.initialize_channel_values(
            &mut vn,
            &channel_llr,
            &mut edges,
            0,
        );
        self.initialize_channel_values(
            &mut vn_comb,
            &channel_llr_comb,
            &mut edges_comb,
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
                let alpha_ij_comb: &QaryLlrs<COMB_SIZE> = check_iter
                    .map(|key| debug_unwrap!(&edges_comb[&key].v2c.as_ref()))
                    .next()
                    .unwrap();

                let mut beta_i = vec![QaryLlrs::<BSIZE>([FloatType::INFINITY; BSIZE]); SW];
                let mut beta_ij_comb = QaryLlrs::<COMB_SIZE>([FloatType::INFINITY; COMB_SIZE]);

                // TODO: actually start supporting less than SW variables
                let mut d_values_iter = SimpleDValueIterator::<BType>::new(BType::from(B).unwrap(), num_nones, SW);
                while let Some(d_values) = d_values_iter.next() {
                    let mut sum_of_alpha: FloatType = alpha_i
                        .iter()
                        .zip(d_values)
                        .map(|(alpha_ij, d)| {
                            alpha_ij.0[C::b2i::<B>(*d)]
                        })
                        .sum();
                    let d_comb_index = C::index_of_comb(d_values);
                    sum_of_alpha += alpha_ij_comb.0[d_comb_index];

                    for ((beta_ij, d), alpha_ij) in beta_i.iter_mut().zip(d_values).zip(&alpha_i) {
                        beta_ij.0[C::b2i::<B>(*d)] = beta_ij.0[C::b2i::<B>(*d)]
                            .min(sum_of_alpha - alpha_ij.0[C::b2i::<B>(*d)]);
                    }
                    beta_ij_comb.0[d_comb_index] = beta_ij_comb.0[d_comb_index]
                        .min(sum_of_alpha - alpha_ij_comb.0[d_comb_index]);
                }

                let mut check_iter = check.variables(check_idx);
                for (key, beta_ij) in check_iter.by_ref().take(num_variable_nodes).zip(&beta_i) {
                    debug_unwrap!(edges.get_mut(&key)).c2v.replace(*beta_ij);
                }
                debug_unwrap!(edges_comb.get_mut(&check_iter.next().unwrap())).c2v.replace(beta_ij_comb);
            }

            // Variable node update (sum)
            for (var_idx, var) in (0..).zip(&*vn) {
                // Collect connected checks

                // 4.1 primitive messages. Full summation
                let mut sum: QaryLlrs<BSIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edges.get(&key)).c2v);
                    sum = P::add_with_mult_by_parity(&sum, &incoming, self.parity_check[&key]);
                }
                for key in var.checks(var_idx) {
                    // 4.2 primitive outgoing messages, subtract self for individual message
                    let edge = debug_unwrap!(edges.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = P::sub_with_mult_by_parity_then_mult(&sum, &incoming, self.parity_check[&key]);
                    // 5. Message normalization
                    let arg_min = Self::arg_min::<BSIZE>(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }

                if it >= self.max_iter {
                    // 6. Tentative decision
                    hard_decision[var_idx as usize] = C::i2b::<B>(Self::arg_min::<BSIZE>(sum));
                }
            }
            for (var_idx, var) in ((BVARS as u16)..).zip(&*vn_comb) {
                let mut sum: QaryLlrs<COMB_SIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx) {
                    let incoming = debug_unwrap!(debug_unwrap!(edges_comb.get(&key)).c2v);
                    sum = P::add_with_mult_by_parity(&sum, &incoming, self.parity_check[&key]);
                }
                for key in var.checks(var_idx) {
                    let edge = debug_unwrap!(edges_comb.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    let prim_out = P::sub_with_mult_by_parity_then_mult(&sum, &incoming, self.parity_check[&key]);
                    let arg_min = Self::arg_min::<COMB_SIZE>(prim_out);
                    let out = prim_out.qary_sub_arg(arg_min);
                    edge.v2c.replace(out);
                }
            }

            if it >= self.max_iter {
                break 'decoding;
            }
        }

        Ok(hard_decision)
    }

    // we return only c2v to B variables since c2v message to combination variables does not
    // change anything since the degree of such variables is 1
    fn check_node_c2v_sum_product(
        check_idx: u16,
        check: &CheckNode,
        edges: &Container2D<Edge<BSIZE>>,
        edges_comb: &Container2D<Edge<COMB_SIZE>>,
        parity_check: &Container2D<ParCheckType>,
        SW: usize,
        ) -> Vec<QaryLlrs<BSIZE>> {
        // Check nodes is a list of values followed by some amount (potentially 0) of Nones
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
        let alpha_ij_comb: &QaryLlrs<COMB_SIZE> = check_iter
            .map(|key| debug_unwrap!(&edges_comb[&key].v2c.as_ref()))
            .next()
            .unwrap();

        // New c2v messages
        let mut beta_i = vec![QaryLlrs::<BSIZE>([FloatType::NEG_INFINITY; BSIZE]); num_variable_nodes];
        
        let mut d_values_iter = SimpleDValueIterator::<BType>::new(BType::from(B).unwrap(), num_nones, SW);
        while let Some(d_values) = d_values_iter.next() {
            let mut sum_of_alpha: FloatType = alpha_i
                .iter()
                .zip(d_values)
                .map(|(alpha_ij, d)| {
                    alpha_ij.0[C::b2i::<B>(*d)]
                })
                .sum();
            let d_comb_index = C::index_of_comb(d_values);
            sum_of_alpha += alpha_ij_comb.0[d_comb_index];

            for ((beta_ij, d), alpha_ij) in beta_i.iter_mut().zip(d_values).zip(&alpha_i) {
                let d_idx = C::b2i::<B>(*d);
                let c2v_value = sum_of_alpha - alpha_ij.0[d_idx];
                beta_ij.0[d_idx] = beta_ij.0[d_idx].ln_add_exp(c2v_value);
            }
        }

        for beta_ij in &mut beta_i[0..num_variable_nodes] {
            normalize_log_probs(&mut beta_ij.0);
        }

        beta_i
    }

    /// Compute |exp(a_log) - exp(b_log)| in a numerically stable way.
    fn log_abs_diff(a_log: FloatType, b_log: FloatType) -> FloatType {
        let (sign, max_log, diff_log) = if a_log > b_log {
            (1.0, a_log, a_log - b_log)
        } else {
            (-1.0, b_log, b_log - a_log)
        };

        if diff_log > 0.0 {
            let log_term = if diff_log < FloatType::ln(2.0) {
                (-((-diff_log).exp_m1())).ln() // log(-expm1(-diff_log))
            } else {
                -diff_log
            };
            (max_log + log_term).exp()
        } else {
            0.0 // a_log == b_log
        }
    }

    // computes l_infty norm over probability domain
    fn log_domain_residual<const T: usize>(old_msg: &QaryLlrs<T>, new_msg: &QaryLlrs<T>) -> FloatType {
        new_msg.0.iter()
            .zip(old_msg.0.iter())
            .map(|(&a, &b)| Self::log_abs_diff(a, b))
            .fold(0.0, FloatType::max)
    }


    // Implements the sum product algorithm
    // Uses IDS scheduler, in particular, Node-wise
    // 
    // Note: the implementation can't handle impossible values, meaning that 
    // channel_llr and channel_llr_comb should not contain -inf. That's because we
    // always assume the algorithm will be used for the imperfect oracle
    pub fn sum_product_nw(&self, channel_llr: Vec<QaryLlrs<BSIZE>>, channel_llr_comb: Vec<QaryLlrs<COMB_SIZE>>) -> Result<(Vec<[FloatType; BSIZE]>, Vec<BType>)> {
        check_all_finite_assert(&channel_llr, "channel_llr");
        check_all_finite_assert(&channel_llr_comb, "channel_llr_comb");

        // Clone the states that we need to mutate
        let mut vn = self.vn.clone();
        let mut vn_comb = self.vn_comb.clone();
        let mut edges = self.edges.clone();
        let mut edges_comb = self.edges_comb.clone();
        
        let BVARS = self.N - self.R; // number of B-variables
        let SW = self.DC - 1; // each check node has SW B-variables + 1 BSUM variable

        // 1, 2: Initialize the channel values (steps are based on https://ieeexplore.ieee.org/abstract/document/5610969)
        let mut c2v_init = QaryLlrs::<BSIZE>([FloatType::default(); BSIZE]);
        normalize_log_probs(&mut c2v_init.0);
        for (var_idx, (v, m)) in (0..).zip(vn.iter_mut().zip(channel_llr)) {
            v.channel = Some(m);
            // For each check node connected to var_idx, copy prior distribution `m` into edges[..].v2c
            for key in v.checks(var_idx) {
                // We assume that only plus or minus ones are present in the parity check matrix
                let mut edge = debug_unwrap!(edges.get_mut(&key));
                edge.v2c.insert(P::mult_by_parity(&m, self.parity_check[&key]));
                edge.c2v.replace(c2v_init.clone());
            }
        }
        // Similarly for comb variables; note: var_idx starts NOT from 0
        for (var_idx, (v, m)) in ((BVARS as Key1D)..).zip(vn_comb.iter_mut().zip(channel_llr_comb)) {
            v.channel = Some(m);
            for key in v.checks(var_idx) {
                let mut edge = debug_unwrap!(edges_comb.get_mut(&key));
                edge.v2c.insert(P::mult_by_parity(&m, self.parity_check[&key]));
            }
        }

        // 3. Compute all \alpha_c
        let mut priorities = vec![FloatType::NEG_INFINITY; self.N];
        for (check_idx, check) in (0..).zip(&*self.cn) {
            // TODO: priority computation can be implemented via min-sum
            let beta_i = Self::check_node_c2v_sum_product(
                check_idx, check, &edges, &edges_comb, &self.parity_check, SW
            );
            let mut alpha_c = FloatType::default();
            for (key, beta_ij) in check.variables(check_idx).zip(&beta_i) {
                let edge = debug_unwrap!(edges.get(&key));
                let residual = Self::log_domain_residual(&c2v_init, beta_ij);
                if (alpha_c < residual) {
                    alpha_c = residual;
                }
            }
            priorities[check_idx as usize] = alpha_c;
        }

        let mut it = 0;
        'decoding: loop {
            it += 1;

            // 4. find i with highest priority
            let mut max_priority_idx = 0;
            // TODO: consider something like priority queue?
            let mut max_priority = FloatType::default();
            for (i, c_priority) in priorities.iter().enumerate() {
                if (*c_priority > max_priority) {
                    max_priority = *c_priority;
                    max_priority_idx = i;
                }
            }

            // 5-12. Note that line 6 with generation and propagation of m_{c_i -> v_k} does NOT
            // depend on messages that are changed in line 9, so, we change computation order a bit

            // 5-6. Compute m_{c_i -> v_k} for all v_k connected to c_i
            let check = &self.cn[max_priority_idx];
            let beta_i = Self::check_node_c2v_sum_product(
                max_priority_idx as Key1D, check, &edges, &edges_comb, &self.parity_check, SW
            );

            // 6. Propagate m_{c_i -> v_k} for variable nodes from check c_i
            for (key, beta_ij) in check.variables(max_priority_idx as Key1D).zip(&beta_i) {
                let mut edge = debug_unwrap!(edges.get_mut(&key));
                edge.c2v.replace(*beta_ij);
            }

            

            // 5-12 \ 6-7. Ignoring lines 6 and 7: For each variable v_k connected to c_i
            // generate and propagate all messages from v_k to check variables different from c_i,
            // then updating priority
            for (key_i, _) in check.variables(max_priority_idx as Key1D).zip(&beta_i) {
                let var_idx = key_i.col();
                let var = &vn[var_idx];

                // 9. Generate and propagate m_{v_k -> c_a}
                let mut sum: QaryLlrs<BSIZE> = debug_unwrap!(var.channel);
                for key in var.checks(var_idx as Key1D) {
                    let edge = debug_unwrap!(edges.get(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    sum = P::add_with_mult_by_parity(&sum, &incoming, self.parity_check[&key]);
                }
                for key in var.checks(var_idx as Key1D) {
                    if (key.row() == max_priority_idx) {
                        continue;
                    }
                    let mut edge = debug_unwrap!(edges.get_mut(&key));
                    let incoming = debug_unwrap!(edge.c2v);
                    // TODO: Very important! if some symbols are impossible, i.e. equal to -inf, then
                    // approach with subtraction is not correct, need to actually compute they "fairly"
                    let mut prim_out = P::sub_with_mult_by_parity_then_mult(&sum, &incoming, self.parity_check[&key]);
                    // message normalization
                    normalize_log_probs(&mut prim_out.0);
                    edge.v2c.replace(prim_out);

                    // 10. compute \alpha_{c_a}
                    let check_idx = key.row();
                    let check_a = &self.cn[check_idx];
                    // TODO: priority computation can be implemented via min-sum
                    let beta_a = Self::check_node_c2v_sum_product(
                        check_idx as Key1D, check_a, &edges, &edges_comb, &self.parity_check, SW
                    );
                    let mut alpha_ca = FloatType::default();
                    for (key_a, beta_aj) in check_a.variables(check_idx as Key1D).zip(&beta_a) {
                        let edge = debug_unwrap!(edges.get(&key_a));
                        let residual = Self::log_domain_residual(&debug_unwrap!(edge.c2v), beta_aj);
                        if (alpha_ca < residual) {
                            alpha_ca = residual;
                        }
                    }
                    priorities[check_idx] = alpha_ca;
                }
            }


            // 7. Set alpha_{c_i} = 0
            priorities[max_priority_idx] = 0.0;
            
            // TODO: implement stopping rule!
            if it >= self.max_iter {
                break 'decoding;
            }

            // println!("END OF ITERATION:");

            // for (var_idx, var) in (0..).zip(&*vn) {
            //     let mut sum: QaryLlrs<BSIZE> = debug_unwrap!(var.channel);
            //     for key in var.checks(var_idx) {
            //         let incoming = debug_unwrap!(debug_unwrap!(edges.get(&key)).c2v);
            //         sum = sum.qary_add_with_mult_in_gf(incoming, self.parity_check[&key]);
            //     }

            //     hard_decision[var_idx as usize] = Self::i2b::<B>(Self::arg_max::<BSIZE>(sum));

            //     println!("var {:?}: {:?} -> {:?}", var_idx, sum, hard_decision[var_idx as usize]);
            // }
            
        }

        let mut hard_decision = vec![BType::zero(); BVARS];
        let mut final_llrs = Vec::with_capacity(BVARS);
        // println!("FINAL DECISION");
        for (var_idx, var) in (0..).zip(&*vn) {
            let mut sum: QaryLlrs<BSIZE> = debug_unwrap!(var.channel);
            for key in var.checks(var_idx) {
                let incoming = debug_unwrap!(debug_unwrap!(edges.get(&key)).c2v);
                sum = P::add_with_mult_by_parity(&sum, &incoming, self.parity_check[&key]);
            }

            final_llrs.push(Self::into_probability_domain(&sum.0));
            hard_decision[var_idx as usize] = C::i2b::<B>(Self::arg_max::<BSIZE>(sum));
        }

        Ok((final_llrs, hard_decision))
    }

    pub fn into_probability_domain<const T: usize>(llr: &[FloatType; T]) -> [FloatType; T] {
        let mut probs = [0.0; T];

        // 1. Find the max value (should be zero ideally, but still useful for stability)
        let mut max_val = FloatType::NEG_INFINITY;
        for &v in llr.iter() {
            if v.is_finite() && v > max_val {
                max_val = v;
            }
        }

        // 2. Compute exp(llr[i] - max) to avoid overflow
        let mut sum = 0.0;
        for (i, &v) in llr.iter().enumerate() {
            let prob = (v - max_val).exp();
            probs[i] = prob;
            sum += prob;
        }

        // 3. Normalize
        if sum == 0.0 {
            probs.fill(1.0 / T as FloatType); // fallback: uniform
        } else {
            for val in probs.iter_mut() {
                *val /= sum;
            }
        }

        probs
    }

    // Convert channel_output probabilities into (shifted) log probabilities.
    //
    // Returns a vector in log domain, where the largest probability
    //   in each vector is mapped to 0.0, and zero-probabilities become -∞.
    //
    // This is appropriate for sum-product decoding.
    pub fn into_log_domain<const T: usize>(channel_output: &Vec<Vec<FloatType>>) -> Vec<QaryLlrs<T>> {
        const EPSILON: FloatType = 0.001;
        let mut llrs: Vec<QaryLlrs<T>> = vec![QaryLlrs::<T>([0.0; T]); channel_output.len()];

        for (var, msg) in channel_output.iter().zip(llrs.iter_mut()) {
            let sum: FloatType = var.iter().sum();
            assert!((1.0 - EPSILON..=1.0 + EPSILON).contains(&sum),
                "Probabilities must sum to ~1.0; got {}", sum);
            let max = var
                .iter()
                .copied()
                // ignore NAN values (errors from the previous line)
                .flat_map(NotNan::new)
                .max()
                .map(NotNan::into_inner)
                .expect("No maximum probability found");

            // convert each probability p to log-domain:
            //    log( p / max ) if p > 0, else -∞
            for (p, dst) in var.iter().zip(msg.0.iter_mut()) {
                if *p <= 0.0 {
                    *dst = FloatType::NEG_INFINITY;
                } else {
                    *dst = (*p / max).ln();
                }
            }
        }

        llrs
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

    fn arg_max<const T: usize>(m: QaryLlrs<T>) -> usize {
        let mut max_val = FloatType::NEG_INFINITY;
        let mut max_arg = 0;
        for (arg, val) in m.0.iter().copied().enumerate() {
            if val > max_val {
                max_val = val;
                max_arg = arg;
            }
        }
        max_arg
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
    use super::*;


    #[test]
    fn system_of_equations_weight_2() {
        // TODO: a bit ugly, but works
        type C = SumCombination<i8, 1>;
        type TestDecoder = DecoderSpecial<C, TernaryParityCheckOps, 1, 3, 5, i8>;

        let mut parity_check = vec![
            vec![1, 1, 0],
            vec![1, 0, 1],
            vec![0, 1, 1],
        ];
        

        let R = parity_check.len();
        let N = parity_check[0].len();
        // Append negative identity matrix to the right
        for (i, row) in parity_check.iter_mut().enumerate() {
            row.extend((0..R).map(|j| if i == j { -1 } else { 0 }));
        }

        let DV = 2;
        let DC = 2+1;
        let num_variables = DC - 1;
        let iterations = 10;
        let decoder = TestDecoder::new(parity_check.clone(), DV, DC, iterations);

        // NTRU distribution
        // let prior_secret: Vec<FloatType> = vec![85.0 / 256.0, 86.0 / 256.0, 85.0 / 256.0];
        let prior_secret: Vec<FloatType> = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let mut channel_output: Vec<Vec<FloatType>> = Vec::with_capacity(N);
        for _ in 0..N {
            channel_output.push(prior_secret.clone());
        }

        let f = vec![0, -1, 1];

        let mut channel_output_comb: Vec<Vec<FloatType>> = Vec::new();
        for row in &parity_check {
            let dot_product: i8 = row.iter().take(N).zip(&f).map(|(a, b)| a * b).sum();
            let mut cond_prob = vec![0.01; TestDecoder::COMB_SIZE];
            
            cond_prob[C::index_of_comb_val(dot_product, num_variables)] = 1.0 - 0.01 * ((TestDecoder::COMB_SIZE - 1) as FloatType);
            channel_output_comb.push(cond_prob);
        }

        let channel_llr = TestDecoder::into_log_domain(&channel_output);
        let channel_llr_comb = TestDecoder::into_log_domain(&channel_output_comb);

        let res = decoder.sum_product_nw(channel_llr, channel_llr_comb).expect("Failed");

        println!("Decoded secret: {:?}", res);

        assert_eq!(res, f);
        // assert!(1 == 0);
    }

    #[test]
    fn system_of_equations_weight_3() {
        type C = SumCombination<i8, 1>;
        type TestDecoder = DecoderSpecial<C, TernaryParityCheckOps, 1, 3, 7, i8>;
        let mut parity_check = vec![
            vec![1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
            vec![0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            vec![0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            vec![1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            vec![0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            vec![1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            vec![0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
            vec![0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        ];
        
        let R = parity_check.len();
        let N = parity_check[0].len();
        // Append negative identity matrix to the right
        for (i, row) in parity_check.iter_mut().enumerate() {
            row.extend((0..R).map(|j| if i == j { -1 } else { 0 }));
        }

        let DV = 4;
        let DC = 3+1;
        let num_variables = DC - 1;
        let iterations = 30;
        let decoder = TestDecoder::new(parity_check.clone(), DV, DC, iterations);

        // NTRU distribution
        // let prior_secret: Vec<FloatType> = vec![85.0 / 256.0, 86.0 / 256.0, 85.0 / 256.0];
        let prior_secret: Vec<FloatType> = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let mut channel_output: Vec<Vec<FloatType>> = Vec::with_capacity(N);
        for _ in 0..N {
            channel_output.push(prior_secret.clone());
        }

        let f = vec![1, 0, 0, 0, 0, 0, -1, 0, 0, 1];

        let mut channel_output_comb: Vec<Vec<FloatType>> = Vec::new();
        for row in &parity_check {
            // considers only 0 or 1 in parity check matrix
            let dot_product: i8 = row.iter().take(N).zip(&f).map(|(a, b)| a * b).sum();
            let mut cond_prob = vec![0.01; TestDecoder::COMB_SIZE];
            cond_prob[C::index_of_comb_val(dot_product, num_variables)] = 1.0 - 0.01 * ((TestDecoder::COMB_SIZE - 1) as FloatType);
            channel_output_comb.push(cond_prob);
        }

        let channel_llr = TestDecoder::into_log_domain(&channel_output);
        let channel_llr_comb = TestDecoder::into_log_domain(&channel_output_comb);

        let res = decoder.sum_product_nw(channel_llr, channel_llr_comb).expect("Failed");

        println!("Decoded secret: {:?}", res);

        assert_eq!(res, f);
        // assert!(1 == 0);
    }

    #[test]
    fn small_kyber_like_test_no_errors() {
        type C = JointCombination<i8, 2, 5>;
        type TestDecoder = DecoderSpecial<C, BinaryParityCheckOps, 2, 5, 25, i8>;
        let parity_check = vec![
            vec![1, 1, 0, 0, 1, 0],
            vec![0, 0, 1, 1, 0, 1],
        ];

        let decoder = TestDecoder::new(parity_check, 1, 3, 7);

        let prior_secret: [FloatType; TestDecoder::BSIZE] = [0.0625, 0.25, 0.375, 0.25, 0.0625];
        let num_variables = 4;
        let mut channel_output: Vec<Vec<FloatType>> = Vec::with_capacity(num_variables);
        for _ in 0..num_variables {
            channel_output.push(prior_secret.to_vec());
        }

        let channel_output_comb: Vec<Vec<FloatType>> = vec![
            {
                let mut cond_prob = vec![0.001; 25];
                cond_prob[0] = 1.0 - 0.001 * 24 as FloatType; // corresponding to (-2, -2)
                cond_prob
            },
            {
                let mut cond_prob = vec![0.001; 25];
                cond_prob[17] = 1.0 - 0.001 * 24 as FloatType; // corresponding to (0, 1)
                cond_prob
            },
        ];

        let channel_llr = TestDecoder::into_log_domain(&channel_output);
        let channel_llr_comb = TestDecoder::into_log_domain(&channel_output_comb);

        let res = decoder.sum_product_nw(channel_llr, channel_llr_comb).expect("Failed");

        let expected: Vec<i8> = vec![-2, -2, 0, 1];

        assert_eq!(res, expected);
    }

}
