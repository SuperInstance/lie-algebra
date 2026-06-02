#![allow(clippy::needless_range_loop, clippy::ptr_arg, clippy::unnecessary_cast)]
use serde::{Deserialize, Serialize};

/// Dense matrix for linear algebra support.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl DenseMatrix {
    pub fn identity(n: usize) -> Self {
        let mut data = vec![vec![0.0; n]; n];
        for i in 0..n {
            data[i][i] = 1.0;
        }
        Self { rows: n, cols: n, data }
    }

    pub fn zero(r: usize, c: usize) -> Self {
        Self { rows: r, cols: c, data: vec![vec![0.0; c]; r] }
    }

    pub fn from_vec(data: Vec<Vec<f64>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { rows, cols, data }
    }

    pub fn multiply(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result[i][j] = sum;
            }
        }
        DenseMatrix::from_vec(result)
    }

    pub fn add(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self.data.iter().zip(&other.data)
            .map(|(r1, r2)| r1.iter().zip(r2).map(|(a, b)| a + b).collect())
            .collect();
        DenseMatrix::from_vec(data)
    }

    pub fn scale(&self, s: f64) -> DenseMatrix {
        let data = self.data.iter()
            .map(|r| r.iter().map(|&v| v * s).collect())
            .collect();
        DenseMatrix::from_vec(data)
    }

    pub fn trace(&self) -> f64 {
        assert_eq!(self.rows, self.cols);
        (0..self.rows).map(|i| self.data[i][i]).sum()
    }

    pub fn determinant(&self) -> f64 {
        assert!(self.rows == self.cols && self.rows <= 6);
        let n = self.rows;
        if n == 0 { return 1.0; }
        if n == 1 { return self.data[0][0]; }
        if n == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }
        let mut det = 0.0;
        for j in 0..n {
            let minor = self.minor(0, j);
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            det += sign * self.data[0][j] * minor.determinant();
        }
        det
    }

    fn minor(&self, row: usize, col: usize) -> DenseMatrix {
        let data: Vec<Vec<f64>> = self.data.iter().enumerate()
            .filter(|(i, _)| *i != row)
            .map(|(_, r)| {
                r.iter().enumerate()
                    .filter(|(j, _)| *j != col)
                    .map(|(_, &v)| v)
                    .collect()
            }).collect();
        DenseMatrix::from_vec(data)
    }

    pub fn transpose(&self) -> DenseMatrix {
        let data = (0..self.cols).map(|j| {
            (0..self.rows).map(|i| self.data[i][j]).collect()
        }).collect();
        DenseMatrix::from_vec(data)
    }

    pub fn commutator(&self, other: &DenseMatrix) -> DenseMatrix {
        self.multiply(other).add(&other.multiply(self).scale(-1.0))
    }

    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i][j] = v;
    }
}

/// A Lie algebra with structure constants.
/// C[i][j][k] is defined so that [e_i, e_j] = Σ_k C[i][j][k] e_k.
/// Structure constants must satisfy antisymmetry C[i][j][k] = -C[j][i][k].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieAlgebra {
    pub name: String,
    pub dimension: usize,
    pub basis: Vec<String>,
    pub structure_constants: Vec<Vec<Vec<f64>>>,
}

impl LieAlgebra {
    pub fn new(name: &str, basis: Vec<String>, structure_constants: Vec<Vec<Vec<f64>>>) -> Self {
        let dim = basis.len();
        Self { name: name.to_string(), dimension: dim, basis, structure_constants }
    }

    /// Compute [Σaᵢeᵢ, Σbⱼeⱼ] via structure constants.
    pub fn bracket(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        assert_eq!(a.len(), self.dimension);
        assert_eq!(b.len(), self.dimension);
        let mut result = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if a[i] == 0.0 || b[j] == 0.0 { continue; }
                for k in 0..self.dimension {
                    result[k] += a[i] * b[j] * self.structure_constants[i][j][k];
                }
            }
        }
        result
    }

    pub fn verify_antisymmetry(&self) -> bool {
        let eps = 1e-10;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    if (self.structure_constants[i][j][k] + self.structure_constants[j][i][k]).abs() > eps {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub fn verify_jacobi(&self) -> bool {
        let eps = 1e-8;
        let n = self.dimension;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    // [eᵢ,[eⱼ,eₖ]] + [eⱼ,[eₖ,eᵢ]] + [eₖ,[eᵢ,eⱼ]] = 0
                    for m in 0..n {
                        let mut sum = 0.0;
                        for l in 0..n {
                            sum += self.structure_constants[i][l][m] * self.structure_constants[j][k][l];
                            sum += self.structure_constants[j][l][m] * self.structure_constants[k][i][l];
                            sum += self.structure_constants[k][l][m] * self.structure_constants[i][j][l];
                        }
                        if sum.abs() > eps {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn is_abelian(&self) -> bool {
        let eps = 1e-10;
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                for k in 0..self.dimension {
                    if self.structure_constants[i][j][k].abs() > eps {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Center: {x : [x,y]=0 for all y}
    pub fn center(&self) -> Vec<Vec<f64>> {
        let eps = 1e-10;
        let n = self.dimension;
        // x is central iff ad(x) = 0 iff Σₖ xₖ C[k][j][i] = 0 for all i,j
        // This gives a system of linear equations for x.
        // Collect all equations: Σₖ xₖ C[k][j][i] = 0
        let mut equations = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let mut eq = vec![0.0; n];
                for k in 0..n {
                    eq[k] = self.structure_constants[k][j][i];
                }
                if eq.iter().any(|&v| v.abs() > eps) {
                    equations.push(eq);
                }
            }
        }
        // Find the null space of this equation system
        row_null_space(&equations, n, eps)
    }

    /// Derived algebra: span of {[x,y]}
    pub fn derived_algebra(&self) -> Vec<Vec<f64>> {
        let eps = 1e-10;
        let n = self.dimension;
        let mut generators = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let br = self.bracket(&unit_vec(i, n), &unit_vec(j, n));
                if br.iter().any(|&v| v.abs() > eps) {
                    generators.push(br);
                }
            }
        }
        row_span(&generators, n, eps)
    }

    /// Is solvable (derived series terminates within max_depth steps)?
    pub fn is_solvable(&self, max_depth: usize) -> bool {
        let eps = 1e-10;
        let n = self.dimension;
        let mut current = self.derived_algebra();
        for _ in 0..max_depth {
            if current.is_empty() || current.iter().all(|v| v.iter().all(|&x| x.abs() < eps)) {
                return true;
            }
            let mut new_gens = Vec::new();
            for a in &current {
                for b in &current {
                    let br = self.bracket(a, b);
                    if br.iter().any(|&v| v.abs() > eps) {
                        new_gens.push(br);
                    }
                }
            }
            let next = row_span(&new_gens, n, eps);
            if next.len() == current.len() {
                let mut combined = current.clone();
                combined.extend(next.clone());
                if row_span(&combined, n, eps).len() == current.len() {
                    return current.is_empty() || current.iter().all(|v| v.iter().all(|&x| x.abs() < eps));
                }
            }
            current = next;
        }
        current.is_empty() || current.iter().all(|v| v.iter().all(|&x| x.abs() < eps))
    }

    /// Is nilpotent (lower central series terminates within max_depth)?
    pub fn is_nilpotent(&self, max_depth: usize) -> bool {
        let eps = 1e-10;
        let n = self.dimension;
        let mut current = self.derived_algebra();
        for _ in 0..max_depth {
            if current.is_empty() || current.iter().all(|v| v.iter().all(|&x| x.abs() < eps)) {
                return true;
            }
            let mut new_gens = Vec::new();
            for i in 0..n {
                let ei = unit_vec(i, n);
                for b in &current {
                    let br = self.bracket(&ei, b);
                    if br.iter().any(|&v| v.abs() > eps) {
                        new_gens.push(br);
                    }
                }
            }
            let next = row_span(&new_gens, n, eps);
            if next.len() == current.len() {
                let mut combined = current.clone();
                combined.extend(next.clone());
                if row_span(&combined, n, eps).len() == current.len() {
                    return current.is_empty() || current.iter().all(|v| v.iter().all(|&x| x.abs() < eps));
                }
            }
            current = next;
        }
        current.is_empty() || current.iter().all(|v| v.iter().all(|&x| x.abs() < eps))
    }

    /// Killing form κ(x,y) = tr(ad(x)∘ad(y))
    pub fn killing_form(&self) -> DenseMatrix {
        let n = self.dimension;
        let mut kf = DenseMatrix::zero(n, n);
        for i in 0..n {
            let ad_i = self.adjoint_matrix(&unit_vec(i, n));
            for j in 0..n {
                let ad_j = self.adjoint_matrix(&unit_vec(j, n));
                let prod = ad_i.multiply(&ad_j);
                kf.set(i, j, prod.trace());
            }
        }
        kf
    }

    /// Cartan's criterion: semisimple iff Killing form is non-degenerate
    pub fn is_semisimple(&self) -> bool {
        let kf = self.killing_form();
        let n = kf.rows;
        if n == 0 { return true; }
        if n <= 6 {
            return kf.determinant().abs() > 1e-8;
        }
        let rows: Vec<Vec<f64>> = kf.data;
        row_span(&rows, n, 1e-8).len() == n
    }

    /// ad(x) as matrix: (ad(x))ᵢⱼ = Σₖ xₖ Cₖⱼᵢ
    /// This means: ad(x) * e_j = Σᵢ (Σₖ xₖ Cₖⱼᵢ) eᵢ = [x, eⱼ]
    pub fn adjoint_matrix(&self, x: &[f64]) -> DenseMatrix {
        let n = self.dimension;
        let mut mat = DenseMatrix::zero(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for k in 0..n {
                    val += x[k] * self.structure_constants[k][j][i];
                }
                mat.set(i, j, val);
            }
        }
        mat
    }
}

// ── Helper: complete antisymmetric structure constants ──

fn antisymmetrize(c: &mut Vec<Vec<Vec<f64>>>) {
    let n = c.len();
    // Collect all non-zero positions first
    let mut entries: Vec<(usize, usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                if c[i][j][k].abs() > 1e-15 {
                    entries.push((i, j, k, c[i][j][k]));
                }
            }
        }
    }
    // Now set antisymmetric counterparts
    for (i, j, k, v) in entries {
        c[j][i][k] = -v;
    }
}

// ── Classical Lie Algebras ──

/// sl(2,ℝ): basis {e,h,f}, [h,e]=2e, [h,f]=-2f, [e,f]=h
pub fn sl2() -> LieAlgebra {
    let n = 3;
    let mut c = vec![vec![vec![0.0; n]; n]; n];
    // [e,f] = h → C[0][2][1] = 1
    c[0][2][1] = 1.0;
    // [h,e] = 2e → C[1][0][0] = 2
    c[1][0][0] = 2.0;
    // [h,f] = -2f → C[1][2][2] = -2
    c[1][2][2] = -2.0;
    antisymmetrize(&mut c);

    LieAlgebra::new("sl(2)", vec!["e".into(), "h".into(), "f".into()], c)
}

/// so(3): [L₁,L₂]=L₃, [L₂,L₃]=L₁, [L₃,L₁]=L₂
pub fn so3() -> LieAlgebra {
    let n = 3;
    let mut c = vec![vec![vec![0.0; n]; n]; n];
    c[0][1][2] = 1.0;
    c[1][2][0] = 1.0;
    c[2][0][1] = 1.0;
    antisymmetrize(&mut c);
    LieAlgebra::new("so(3)", vec!["L1".into(), "L2".into(), "L3".into()], c)
}

/// Heisenberg algebra of dimension 2n+1
pub fn heisenberg(n: usize) -> LieAlgebra {
    let dim = 2 * n + 1;
    let mut basis_names = Vec::new();
    for i in 0..n { basis_names.push(format!("x{}", i)); }
    for i in 0..n { basis_names.push(format!("y{}", i)); }
    basis_names.push("z".into());

    let mut c = vec![vec![vec![0.0; dim]; dim]; dim];
    // [xᵢ, yⱼ] = δᵢⱼ z
    for i in 0..n {
        c[i][n + i][dim - 1] = 1.0;
    }
    antisymmetrize(&mut c);
    LieAlgebra::new(&format!("heisenberg({})", n), basis_names, c)
}

/// gl(n) — general linear Lie algebra
pub fn gl(n: usize) -> LieAlgebra {
    let dim = n * n;
    let mut basis_names = Vec::new();
    for i in 0..n {
        for j in 0..n {
            basis_names.push(format!("E{}{}", i, j));
        }
    }
    let mut c = vec![vec![vec![0.0; dim]; dim]; dim];
    for ij in 0..dim {
        let i = ij / n;
        let j = ij % n;
        for kl in 0..dim {
            let k = kl / n;
            let l = kl % n;
            if j == k {
                let il = i * n + l;
                c[ij][kl][il] += 1.0;
            }
            if l == i {
                let kj = k * n + j;
                c[ij][kl][kj] -= 1.0;
            }
        }
    }
    LieAlgebra::new(&format!("gl({})", n), basis_names, c)
}

/// Strictly upper triangular matrices — nilpotent
pub fn upper_triangular(n: usize) -> LieAlgebra {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    let dim = pairs.len();
    let basis_names: Vec<String> = pairs.iter().map(|(i, j)| format!("E{}{}", i, j)).collect();
    let mut idx_map = std::collections::HashMap::new();
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        idx_map.insert((i, j), idx);
    }

    let mut c = vec![vec![vec![0.0; dim]; dim]; dim];
    for a in 0..dim {
        let (i, j) = pairs[a];
        for b in 0..dim {
            let (k, l) = pairs[b];
            if j == k {
                if let Some(&idx) = idx_map.get(&(i, l)) {
                    c[a][b][idx] += 1.0;
                }
            }
            if l == i {
                if let Some(&idx) = idx_map.get(&(k, j)) {
                    c[a][b][idx] -= 1.0;
                }
            }
        }
    }
    LieAlgebra::new(&format!("upper_triangular({})", n), basis_names, c)
}

/// ℝ³ with cross product bracket: isomorphic to so(3)
pub fn cross_product_lie() -> LieAlgebra {
    let n = 3;
    let mut c = vec![vec![vec![0.0; n]; n]; n];
    c[0][1][2] = 1.0;
    c[1][2][0] = 1.0;
    c[2][0][1] = 1.0;
    antisymmetrize(&mut c);
    LieAlgebra::new("cross_product(R³)", vec!["e1".into(), "e2".into(), "e3".into()], c)
}

// ── LieRepresentation ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LieRepresentation {
    pub algebra: LieAlgebra,
    pub dimension: usize,
    pub rho: Vec<DenseMatrix>,
}

impl LieRepresentation {
    pub fn new(algebra: LieAlgebra, dimension: usize, rho: Vec<DenseMatrix>) -> Self {
        assert_eq!(rho.len(), algebra.dimension);
        for m in &rho {
            assert_eq!(m.rows, dimension);
            assert_eq!(m.cols, dimension);
        }
        Self { algebra, dimension, rho }
    }

    /// ρ([X,Y]) = ρ(X)ρ(Y) - ρ(Y)ρ(X)
    pub fn verify_homomorphism(&self) -> bool {
        let eps = 1e-8;
        let n = self.algebra.dimension;
        for i in 0..n {
            for j in 0..n {
                let lhs = self.rho[i].commutator(&self.rho[j]);
                // Compute ρ([eᵢ,eⱼ])
                let mut rhs = DenseMatrix::zero(self.dimension, self.dimension);
                for k in 0..n {
                    let c = self.algebra.structure_constants[i][j][k];
                    if c.abs() > eps {
                        rhs = rhs.add(&self.rho[k].scale(c));
                    }
                }
                for r in 0..self.dimension {
                    for c in 0..self.dimension {
                        if (lhs.data[r][c] - rhs.data[r][c]).abs() > eps {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    pub fn character(&self, element: &[f64]) -> f64 {
        let mut mat = DenseMatrix::zero(self.dimension, self.dimension);
        for (i, &coeff) in element.iter().enumerate() {
            mat = mat.add(&self.rho[i].scale(coeff));
        }
        mat.trace()
    }

    pub fn is_irreducible(&self) -> bool {
        if self.dimension > 6 { return true; }
        let n = self.dimension;
        let eps = 1e-8;
        for seed in 0..n {
            let mut generators = vec![unit_vec(seed, n)];
            let mut changed = true;
            while changed {
                changed = false;
                let current_span = row_span(&generators, n, eps);
                for rho_i in &self.rho {
                    for v in &current_span {
                        let mut new_v = vec![0.0; n];
                        for r in 0..n {
                            for cc in 0..n {
                                new_v[r] += rho_i.data[r][cc] * v[cc];
                            }
                        }
                        if new_v.iter().any(|&x| x.abs() > eps) {
                            let mut test = generators.clone();
                            test.push(new_v);
                            let expanded = row_span(&test, n, eps);
                            if expanded.len() > current_span.len() {
                                generators = test;
                                changed = true;
                                break;
                            }
                        }
                    }
                    if changed { break; }
                }
            }
            let span = row_span(&generators, n, eps);
            let dim = span.len();
            if dim > 0 && dim < n {
                return false;
            }
        }
        true
    }

    pub fn direct_sum(&self, other: &LieRepresentation) -> LieRepresentation {
        assert_eq!(self.algebra.dimension, other.algebra.dimension);
        let new_dim = self.dimension + other.dimension;
        let rho: Vec<DenseMatrix> = self.rho.iter().zip(&other.rho).map(|(a, b)| {
            let mut m = DenseMatrix::zero(new_dim, new_dim);
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    m.data[i][j] = a.data[i][j];
                }
            }
            for i in 0..other.dimension {
                for j in 0..other.dimension {
                    m.data[self.dimension + i][self.dimension + j] = b.data[i][j];
                }
            }
            m
        }).collect();
        LieRepresentation::new(self.algebra.clone(), new_dim, rho)
    }

    pub fn tensor_product(&self, other: &LieRepresentation) -> LieRepresentation {
        assert_eq!(self.algebra.dimension, other.algebra.dimension);
        let new_dim = self.dimension * other.dimension;
        let rho: Vec<DenseMatrix> = self.rho.iter().zip(&other.rho).map(|(a, b)| {
            let i1 = DenseMatrix::identity(self.dimension);
            let i2 = DenseMatrix::identity(other.dimension);
            let term1 = kronecker(a, &i2);
            let term2 = kronecker(&i1, b);
            term1.add(&term2)
        }).collect();
        LieRepresentation::new(self.algebra.clone(), new_dim, rho)
    }
}

fn kronecker(a: &DenseMatrix, b: &DenseMatrix) -> DenseMatrix {
    let r = a.rows * b.rows;
    let c = a.cols * b.cols;
    let mut data = vec![vec![0.0; c]; r];
    for i1 in 0..a.rows {
        for j1 in 0..a.cols {
            for i2 in 0..b.rows {
                for j2 in 0..b.cols {
                    data[i1 * b.rows + i2][j1 * b.cols + j2] = a.data[i1][j1] * b.data[i2][j2];
                }
            }
        }
    }
    DenseMatrix::from_vec(data)
}

// ── ExponentialMap ──

pub struct ExponentialMap;

impl ExponentialMap {
    pub fn exp_of(algebra: &LieAlgebra, x: &[f64], terms: usize) -> DenseMatrix {
        let ad_x = algebra.adjoint_matrix(x);
        let n = algebra.dimension;
        let mut result = DenseMatrix::identity(n);
        let mut power = DenseMatrix::identity(n);
        let mut factorial = 1.0;
        for k in 1..=terms {
            power = power.multiply(&ad_x);
            factorial *= k as f64;
            result = result.add(&power.scale(1.0 / factorial));
        }
        result
    }

    /// BCH formula to order 3
    pub fn baker_campbell_hausdorff(algebra: &LieAlgebra, x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = algebra.dimension;
        let xy = algebra.bracket(x, y);
        let x_xy = algebra.bracket(x, &xy);
        let y_xy = algebra.bracket(y, &xy);

        let mut z = vec![0.0; n];
        for i in 0..n {
            z[i] = x[i] + y[i]
                + 0.5 * xy[i]
                + x_xy[i] / 12.0
                - y_xy[i] / 12.0;
        }
        z
    }
}

// ── Root Systems & Dynkin ──

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootSystem {
    pub roots: Vec<Vec<f64>>,
    pub simple_roots: Vec<Vec<f64>>,
}

impl RootSystem {
    pub fn cartan_matrix(&self) -> DenseMatrix {
        let n = self.simple_roots.len();
        let mut cm = DenseMatrix::zero(n, n);
        for i in 0..n {
            for j in 0..n {
                let dot = dot_product(&self.simple_roots[i], &self.simple_roots[j]);
                let norm_i = dot_product(&self.simple_roots[i], &self.simple_roots[i]);
                cm.set(i, j, 2.0 * dot / norm_i);
            }
        }
        cm
    }

    pub fn dynkin_diagram(&self) -> DynkinDiagram {
        let cm = self.cartan_matrix();
        let n = self.simple_roots.len();
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let val = cm.data[i][j].round() as i32;
                if val != 0 {
                    edges.push((i, j, val.unsigned_abs() as u32));
                }
            }
        }
        DynkinDiagram { nodes: n, edges }
    }

    pub fn positive_roots(&self) -> Vec<Vec<f64>> {
        // Positive roots: those where the first nonzero component is positive
        let eps = 1e-10;
        self.roots.iter()
            .filter(|r| {
                for &v in r.iter() {
                    if v.abs() > eps {
                        return v > eps;
                    }
                }
                false
            })
            .cloned()
            .collect()
    }

    pub fn weyl_group_size(&self) -> usize {
        let dd = self.dynkin_diagram();
        match dd.classify() {
            DynkinType::A(n) => (1..=n + 1).product(),
            DynkinType::B(n) | DynkinType::C(n) => (1..=n).product::<usize>() * 2usize.pow(n as u32),
            DynkinType::D(n) => (1..=n).product::<usize>() * 2usize.pow((n - 1) as u32),
            DynkinType::E(6) => 51840,
            DynkinType::E(7) => 2903040,
            DynkinType::E(8) => 696729600,
            DynkinType::F4 => 1152,
            DynkinType::G2 => 12,
            DynkinType::E(_) => 0,
            DynkinType::Unknown => 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynkinDiagram {
    pub nodes: usize,
    pub edges: Vec<(usize, usize, u32)>,
}

impl DynkinDiagram {
    pub fn classify(&self) -> DynkinType {
        let n = self.nodes;
        if self.edges.is_empty() && n == 1 {
            return DynkinType::A(1);
        }
        let edge_count = self.edges.len();

        // G₂: 2 nodes, one edge with multiplicity 3
        if n == 2 && edge_count == 1 && self.edges[0].2 == 3 {
            return DynkinType::G2;
        }

        // F₄: 4 nodes, chain with edge pattern 1-2-1 (must check before Bₙ)
        if n == 4 && edge_count == 3 {
            let mut mults: Vec<u32> = self.edges.iter().map(|(_, _, m)| *m).collect();
            mults.sort();
            if mults == vec![1, 1, 2] && is_chain(n, &self.edges) {
                return DynkinType::F4;
            }
        }

        // Bₙ: one edge with multiplicity 2, rest multiplicity 1, chain structure
        if n >= 2 {
            let m2_count = self.edges.iter().filter(|(_, _, m)| *m == 2).count();
            let m1_count = self.edges.iter().filter(|(_, _, m)| *m == 1).count();
            if m2_count == 1 && m1_count == n - 2 && edge_count == n - 1 && is_chain(n, &self.edges) {
                return DynkinType::B(n);
            }
        }

        // Simple chain (all multiplicity 1) → Aₙ
        if self.edges.iter().all(|(_, _, m)| *m == 1) && edge_count == n - 1 && is_chain(n, &self.edges) {
            return DynkinType::A(n);
        }

        // Dₙ: Y-shaped with all multiplicity 1
        if n >= 4 && self.edges.iter().all(|(_, _, m)| *m == 1) && edge_count == n - 1 {
            let mut deg = vec![0usize; n];
            for (i, j, _) in &self.edges {
                deg[*i] += 1;
                deg[*j] += 1;
            }
            let deg3 = deg.iter().filter(|&&d| d == 3).count();
            let deg1 = deg.iter().filter(|&&d| d == 1).count();
            if deg3 == 1 && deg1 == 3 {
                return DynkinType::D(n);
            }
        }

        DynkinType::Unknown
    }
}

fn is_chain(n: usize, edges: &[(usize, usize, u32)]) -> bool {
    if edges.len() != n - 1 { return false; }
    let mut deg = vec![0usize; n];
    for (i, j, _) in edges {
        deg[*i] += 1;
        deg[*j] += 1;
    }
    let endpoints = deg.iter().filter(|&&d| d == 1).count();
    let internal = deg.iter().filter(|&&d| d == 2).count();
    endpoints == 2 && internal == n - 2
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DynkinType {
    A(usize),
    B(usize),
    C(usize),
    D(usize),
    E(u8),
    F4,
    G2,
    Unknown,
}

// ── Root system constructors ──

pub fn sl2_root_system() -> RootSystem {
    // sl(2) has one simple root α = [1, -1] in R²
    // Positive root: α, negative root: -α
    let simple = vec![vec![1.0, -1.0]];
    let roots = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
    RootSystem { roots, simple_roots: simple }
}

pub fn sl_n_root_system(n: usize) -> RootSystem {
    let mut simple = Vec::new();
    for i in 0..n - 1 {
        let mut alpha = vec![0.0; n];
        alpha[i] = 1.0;
        alpha[i + 1] = -1.0;
        simple.push(alpha);
    }
    let mut roots = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let mut r = vec![0.0; n];
                r[i] = 1.0;
                r[j] = -1.0;
                roots.push(r);
            }
        }
    }
    RootSystem { roots, simple_roots: simple }
}

pub fn sl2_fundamental_representation() -> LieRepresentation {
    let alg = sl2();
    let rho_e = DenseMatrix::from_vec(vec![vec![0.0, 1.0], vec![0.0, 0.0]]);
    let rho_h = DenseMatrix::from_vec(vec![vec![1.0, 0.0], vec![0.0, -1.0]]);
    let rho_f = DenseMatrix::from_vec(vec![vec![0.0, 0.0], vec![1.0, 0.0]]);
    LieRepresentation::new(alg, 2, vec![rho_e, rho_h, rho_f])
}

// ── Helpers ──

pub(crate) fn unit_vec(i: usize, n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[i] = 1.0;
    v
}

fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn row_span(vectors: &[Vec<f64>], n: usize, eps: f64) -> Vec<Vec<f64>> {
    if vectors.is_empty() { return vec![]; }
    let mut mat: Vec<Vec<f64>> = vectors.to_vec();
    let mut row = 0;
    for col in 0..n {
        let mut found = None;
        for r in row..mat.len() {
            if mat[r][col].abs() > eps {
                found = Some(r);
                break;
            }
        }
        if let Some(pivot_row) = found {
            mat.swap(row, pivot_row);
            let scale = mat[row][col];
            for j in 0..n {
                mat[row][j] /= scale;
            }
            for r in 0..mat.len() {
                if r != row && mat[r][col].abs() > eps {
                    let factor = mat[r][col];
                    for j in 0..n {
                        mat[r][j] -= factor * mat[row][j];
                    }
                }
            }
            row += 1;
        }
    }
    mat.into_iter().filter(|r| r.iter().any(|&v| v.abs() > eps)).collect()
}

/// Find null space of a system of equations (rows are equations, n is the number of variables)
fn row_null_space(equations: &[Vec<f64>], n: usize, eps: f64) -> Vec<Vec<f64>> {
    if equations.is_empty() {
        return (0..n).map(|i| unit_vec(i, n)).collect();
    }
    // Augmented matrix: rows are equations, cols are variables
    // Row reduce, then identify free variables and construct null space basis
    let mut mat: Vec<Vec<f64>> = equations.to_vec();
    let num_eq = mat.len();
    let mut pivot_cols: Vec<usize> = Vec::new();
    let mut row = 0;
    for col in 0..n {
        if row >= num_eq { break; }
        let mut found = None;
        for r in row..num_eq {
            if mat[r][col].abs() > eps {
                found = Some(r);
                break;
            }
        }
        if let Some(pivot_row) = found {
            mat.swap(row, pivot_row);
            let scale = mat[row][col];
            for j in 0..n {
                mat[row][j] /= scale;
            }
            for r in 0..num_eq {
                if r != row && mat[r][col].abs() > eps {
                    let factor = mat[r][col];
                    for j in 0..n {
                        mat[r][j] -= factor * mat[row][j];
                    }
                }
            }
            pivot_cols.push(col);
            row += 1;
        }
    }
    // pivot_cols are the pivot columns; free variables are the rest
    let pivot_set: std::collections::HashSet<usize> = pivot_cols.iter().copied().collect();
    let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_set.contains(c)).collect();
    // For each free variable, construct a null space vector
    let mut basis = Vec::new();
    for fc in &free_cols {
        let mut v = vec![0.0; n];
        v[*fc] = 1.0;
        // Set pivot variables from row-reduced equations
        for (i, &pc) in pivot_cols.iter().enumerate() {
            v[pc] = -mat[i][*fc];
        }
        basis.push(v);
    }
    basis
}

#[cfg(test)]
mod tests;
