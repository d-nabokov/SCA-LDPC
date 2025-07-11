/// Use this macro to create new decoders for different sizes/parameters
///
/// # Usage:
///
/// register_py_decoder_class!(module <= Name{
///     N: <number of variable nodes>,
///     R: <number of check nodes>,
///     DV: <Maximum variable node degree (num checks, per variable)>,
///     DC: <Maximum check node degree (num variables, per check)>,
///     GF: Galois Field to operate on. ("Q" submessages)
/// });
macro_rules! register_py_decoder_class {
    ($m:ident <= $Name:ident{N: $N:literal, R: $R:literal, DV: $DV:literal, DC: $DC:literal, B: $B:literal}) => {{
        type CustomDecoder = Decoder<$N, $R, $DV, $DC, {$B * 2 + 1}, $B, i8>;

        #[pyclass]
        struct $Name {
            decoder: CustomDecoder,
        }

        #[pymethods]
        impl $Name {
            #[new]
            fn new(py_parity_check: PyReadonlyArray2<i8>, iterations: u32) -> Result<Self> {
                let py_parity_check = py_parity_check.as_array();
                ::log::info!(
                    "Constructing decoder {} with N={N}, R={R}, DV={DV}, DC={DC}, GF={GF}, Input parity check matrix has the shape: {shape:?}",
                    stringify!($Name),
                    N = $N,
                    R = $R,
                    DV = $DV,
                    DC = $DC,
                    GF = stringify!($GF),
                    shape = py_parity_check.shape()
                );
                let mut parity_check = [[0; $N]; $R];
                for row in 0..parity_check.len() {
                    for col in 0..parity_check[row].len() {
                        parity_check[row][col] = py_parity_check[(row, col)];
                    }
                }
                Ok($Name {
                    decoder: Decoder::new(parity_check, iterations),
                })
            }

            /// min_sum algorithm
            ///
            /// This method is parallelizable from python.
            ///
            /// Attempts have  been made to make this function parallel from within,
            /// but that resulted in performance loss
            fn min_sum(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>) -> Result<[i8; $N]> {
                let py_channel_output = py_channel_output.as_array();
                py.allow_threads(||{
                    let mut channel_output = [[0.0; {$B * 2 + 1}]; $N];
                    for variable in 0..channel_output.len() {
                        for value in 0..channel_output[variable].len() {
                            channel_output[variable][value] = py_channel_output[(variable, value)].into();
                        }
                    }
                    let channel_llr = CustomDecoder::into_llr(&channel_output);
                    self.decoder.min_sum(channel_llr)
                })
            }
        }

        $m.add_class::<$Name>()?;
    }};
}


/// Use this macro to create new special q-ary decoders for different sizes/parameters
///
/// # Usage:
///
/// register_py_decoder_joint_distribution_class!(module <= Name{
///     B: <first N-R variables assumed to have values from the range [-B, ..., 0, ..., B]>
///     SW: <Maximum number of B-variables connected to check node>,
/// });
/// where N - number of columns, R - number of rows in the parity check matrix
#[macro_export]
macro_rules! __register_py_decoder_special_impl {
    (
        $m:ident,
        $Name:ident,
        B = $B:literal,
        SW = $SW:literal,
        COMB_SIZE = $comb_size:expr,
        COMB_OPS = $comb_ops:ty,
        PARITY_OPS = $parity_ops:ty,
    ) => {{
        const BSIZE: usize = $B * 2 + 1;
        const COMB_SIZE: usize = $comb_size;
        type CustomDecoder = DecoderSpecial<
            $comb_ops,
            $parity_ops,
            $B,
            BSIZE, 
            COMB_SIZE, 
            i8
        >;

        #[pyclass]
        struct $Name {
            decoder: CustomDecoder,
        }

        #[pymethods]
        impl $Name {
            #[new]
            fn new(py_parity_check: PyReadonlyArray2<i8>, DV: usize, DC: usize, iterations: u32) -> Result<Self> {
                let py_parity_check = py_parity_check.as_array();
                let (R, _) = py_parity_check.dim();

                let mut parity_check: Vec<Vec<i8>> = Vec::with_capacity(R);
                for row in py_parity_check.outer_iter() {
                    parity_check.push(row.to_vec());
                }
                Ok($Name {
                    decoder: DecoderSpecial::new(parity_check, DV, DC, iterations),
                })
            }

            /// min_sum algorithm
            ///
            /// This method is parallelizable from python.
            ///
            /// Attempts have  been made to make this function parallel from within,
            /// but that resulted in performance loss
            fn min_sum(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_comb: PyReadonlyArray2<FloatType>) -> Result<Vec<i8>> {
                let py_channel_output = py_channel_output.as_array();
                let py_channel_output_comb = py_channel_output_comb.as_array();
                let R = py_channel_output_comb.dim().0;
                let SV = py_channel_output.dim().0;
                py.allow_threads(||{
                    let mut channel_output: Vec<Vec<FloatType>> = vec![vec![0.0; 2 * $B + 1]; SV];
                    let mut channel_output_comb: Vec<Vec<FloatType>> = vec![vec![0.0; COMB_SIZE]; R];

                    for (variable, output_row) in channel_output.iter_mut().enumerate() {
                        for (value, output_value) in output_row.iter_mut().enumerate() {
                            *output_value = py_channel_output[(variable, value)].into();
                        }
                    }

                    for (variable, output_comb_row) in channel_output_comb.iter_mut().enumerate() {
                        for (value, output_comb_value) in output_comb_row.iter_mut().enumerate() {
                            *output_comb_value = py_channel_output_comb[(variable, value)].into();
                        }
                    }
                    let channel_llr = CustomDecoder::into_llr(&channel_output);
                    let channel_llr_comb = CustomDecoder::into_llr(&channel_output_comb);
                    self.decoder.min_sum(channel_llr, channel_llr_comb)
                })
            }

            /// sum product algorithm
            fn _decode_impl(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_comb: PyReadonlyArray2<FloatType>, scheduler_option: u8) -> Result<(Vec<[FloatType; BSIZE]>, Vec<i8>)> {
                let py_channel_output = py_channel_output.as_array();
                let py_channel_output_comb = py_channel_output_comb.as_array();
                let R = py_channel_output_comb.dim().0;
                let SV = py_channel_output.dim().0;
                py.allow_threads(||{
                    let mut channel_output: Vec<Vec<FloatType>> = vec![vec![0.0; 2 * $B + 1]; SV];
                    let mut channel_output_comb: Vec<Vec<FloatType>> = vec![vec![0.0; COMB_SIZE]; R];

                    for (variable, output_row) in channel_output.iter_mut().enumerate() {
                        for (value, output_value) in output_row.iter_mut().enumerate() {
                            *output_value = py_channel_output[(variable, value)].into();
                        }
                    }

                    for (variable, output_comb_row) in channel_output_comb.iter_mut().enumerate() {
                        for (value, output_comb_value) in output_comb_row.iter_mut().enumerate() {
                            *output_comb_value = py_channel_output_comb[(variable, value)].into();
                        }
                    }
                    let channel_llr = CustomDecoder::into_log_domain(&channel_output);
                    let channel_llr_comb = CustomDecoder::into_log_domain(&channel_output_comb);
                    match scheduler_option {
                        // TODO: quite a bad code, should replace by Enum, had some
                        // problems with visibility of Enum
                        0 => {
                            self.decoder.sum_product_nw(channel_llr, channel_llr_comb)
                        }
                        _ => {
                            self.decoder.sum_product_layered(channel_llr, channel_llr_comb)
                        }
                    }
                })
            }

            fn decode_hard(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_comb: PyReadonlyArray2<FloatType>) -> Result<Vec<i8>> {
                let (_, hard_decision) = self._decode_impl(py, py_channel_output, py_channel_output_comb, 0)?;
                Ok(hard_decision)
            }

            fn decode_with_pr(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_comb: PyReadonlyArray2<FloatType>) -> Result<Vec<[FloatType; BSIZE]>> {
                let (probs, _) = self._decode_impl(py, py_channel_output, py_channel_output_comb, 0)?;
                Ok(probs)
            }

            fn decode_hard_layered(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_comb: PyReadonlyArray2<FloatType>) -> Result<Vec<i8>> {
                let (_, hard_decision) = self._decode_impl(py, py_channel_output, py_channel_output_comb, 1)?;
                Ok(hard_decision)
            }

            fn decode_with_pr_layered(&self, py: Python<'_>, py_channel_output: PyReadonlyArray2<FloatType>, py_channel_output_comb: PyReadonlyArray2<FloatType>) -> Result<Vec<[FloatType; BSIZE]>> {
                let (probs, _) = self._decode_impl(py, py_channel_output, py_channel_output_comb, 1)?;
                Ok(probs)
            }
        }

        $m.add_class::<$Name>()?;
    }};
}

// working over joint distribution, parity check is binary
macro_rules! register_py_decoder_joint_distribution_class {
    ($m:ident <= $Name:ident { B: $B:literal, SW: $SW:literal }) => {{
        $crate::__register_py_decoder_special_impl!(
            $m,
            $Name,
            B = $B,
            SW = $SW,
            COMB_SIZE = BSIZE.pow($SW as u32),
            COMB_OPS = JointCombination<i8, $B, BSIZE>,
            PARITY_OPS = BinaryParityCheckOps,
        );
    }};
}

// working over sums, parity check is ternary
macro_rules! register_py_decoder_sum_distribution_class {
    ($m:ident <= $Name:ident { B: $B:literal, SW: $SW:literal }) => {{
        $crate::__register_py_decoder_special_impl!(
            $m,
            $Name,
            B = $B,
            SW = $SW,
            COMB_SIZE = (($SW as usize) * 2 * $B) + 1,
            COMB_OPS = SumCombination<i8, $B>,
            PARITY_OPS = TernaryParityCheckOps,
        );
    }};
}