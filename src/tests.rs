use super::*;

const EPS: f64 = 1e-8;

// ── DenseMatrix tests ──

#[test]
fn test_identity() {
    let m = DenseMatrix::identity(3);
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(1, 1), 1.0);
    assert_eq!(m.get(0, 1), 0.0);
}

#[test]
fn test_zero_matrix() {
    let m = DenseMatrix::zero(2, 3);
    assert_eq!(m.rows, 2);
    assert_eq!(m.cols, 3);
    assert!(m.data.iter().all(|r| r.iter().all(|&v| v == 0.0)));
}

#[test]
fn test_multiply_identity() {
    let a = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let i = DenseMatrix::identity(2);
    assert_eq!(a.multiply(&i), a);
}

#[test]
fn test_multiply_2x2() {
    let a = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = DenseMatrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = a.multiply(&b);
    assert_eq!(c.data[0][0], 19.0);
    assert_eq!(c.data[0][1], 22.0);
    assert_eq!(c.data[1][0], 43.0);
    assert_eq!(c.data[1][1], 50.0);
}

#[test]
fn test_add() {
    let a = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = DenseMatrix::from_vec(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = a.add(&b);
    assert_eq!(c.data[0][0], 6.0);
    assert_eq!(c.data[1][1], 12.0);
}

#[test]
fn test_scale() {
    let a = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let c = a.scale(2.0);
    assert_eq!(c.data[0][0], 2.0);
    assert_eq!(c.data[1][1], 8.0);
}

#[test]
fn test_trace() {
    let m = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(m.trace(), 5.0);
}

#[test]
fn test_determinant_2x2() {
    let m = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert!((m.determinant() - (-2.0)).abs() < EPS);
}

#[test]
fn test_determinant_3x3() {
    let m = DenseMatrix::from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 0.0]]);
    assert!((m.determinant() - 27.0).abs() < EPS);
}

#[test]
fn test_determinant_identity() {
    let m = DenseMatrix::identity(4);
    assert!((m.determinant() - 1.0).abs() < EPS);
}

#[test]
fn test_transpose() {
    let m = DenseMatrix::from_vec(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let t = m.transpose();
    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);
    assert_eq!(t.data[0][0], 1.0);
    assert_eq!(t.data[0][1], 4.0);
    assert_eq!(t.data[1][0], 2.0);
    assert_eq!(t.data[2][0], 3.0);
    assert_eq!(t.data[2][1], 6.0);
}

#[test]
fn test_commutator() {
    let a = DenseMatrix::from_vec(vec![vec![0.0, 1.0], vec![0.0, 0.0]]);
    let b = DenseMatrix::from_vec(vec![vec![0.0, 0.0], vec![1.0, 0.0]]);
    let c = a.commutator(&b);
    assert_eq!(c.data[0][0], 1.0);
    assert_eq!(c.data[1][1], -1.0);
}

#[test]
fn test_matrix_set_get() {
    let mut m = DenseMatrix::zero(2, 2);
    m.set(0, 1, 42.0);
    assert_eq!(m.get(0, 1), 42.0);
}

#[test]
fn test_serde_matrix() {
    let m = DenseMatrix::from_vec(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let json = serde_json::to_string(&m).unwrap();
    let m2: DenseMatrix = serde_json::from_str(&json).unwrap();
    assert_eq!(m, m2);
}

// ── sl(2) tests ──

#[test]
fn test_sl2_brackets() {
    let alg = sl2();
    // basis: e=0, h=1, f=2
    // [h,e] = 2e
    let he = alg.bracket(&[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]);
    assert!((he[0] - 2.0).abs() < EPS);
    assert!(he[1].abs() < EPS);
    assert!(he[2].abs() < EPS);

    // [h,f] = -2f
    let hf = alg.bracket(&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]);
    assert!(hf[0].abs() < EPS);
    assert!(hf[1].abs() < EPS);
    assert!((hf[2] - (-2.0)).abs() < EPS);

    // [e,f] = h
    let ef = alg.bracket(&[1.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);
    assert!(ef[0].abs() < EPS);
    assert!((ef[1] - 1.0).abs() < EPS);
    assert!(ef[2].abs() < EPS);
}

#[test]
fn test_sl2_antisymmetry() {
    assert!(sl2().verify_antisymmetry());
}

#[test]
fn test_sl2_jacobi() {
    assert!(sl2().verify_jacobi());
}

#[test]
fn test_sl2_not_abelian() {
    assert!(!sl2().is_abelian());
}

#[test]
fn test_sl2_semisimple() {
    assert!(sl2().is_semisimple());
}

#[test]
fn test_sl2_center_empty() {
    assert_eq!(sl2().center().len(), 0);
}

#[test]
fn test_sl2_killing_form() {
    let kf = sl2().killing_form();
    // For sl(2) in basis {e,h,f}, the Killing form κ(x,y) = tr(ad(x)·ad(y)):
    // κ(e,e) = 0, κ(f,f) = 0, κ(h,h) = 8
    // κ(e,f) = κ(f,e) = 4, κ(e,h) = κ(h,f) = 0
    assert!((kf.data[0][0] - 0.0).abs() < EPS);
    assert!((kf.data[2][2] - 0.0).abs() < EPS);
    assert!((kf.data[1][1] - 8.0).abs() < EPS);
    assert!((kf.data[0][2] - 4.0).abs() < EPS, "κ(e,f) = {}", kf.data[0][2]);
    assert!((kf.data[2][0] - 4.0).abs() < EPS);
    assert!(kf.data[0][1].abs() < EPS);
    assert!(kf.data[1][0].abs() < EPS);
}

#[test]
fn test_sl2_not_solvable() {
    assert!(!sl2().is_solvable(10));
}

#[test]
fn test_sl2_not_nilpotent() {
    assert!(!sl2().is_nilpotent(10));
}

#[test]
fn test_sl2_adjoint_matrix() {
    let alg = sl2();
    let ad_h = alg.adjoint_matrix(&[0.0, 1.0, 0.0]);
    assert!((ad_h.data[0][0] - 2.0).abs() < EPS);
    assert!((ad_h.data[2][2] - (-2.0)).abs() < EPS);
}

// ── so(3) tests ──

#[test]
fn test_so3_brackets() {
    let alg = so3();
    let r = alg.bracket(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
    assert!((r[2] - 1.0).abs() < EPS);
}

#[test]
fn test_so3_jacobi() {
    assert!(so3().verify_jacobi());
}

#[test]
fn test_so3_antisymmetry() {
    assert!(so3().verify_antisymmetry());
}

#[test]
fn test_so3_semisimple() {
    assert!(so3().is_semisimple());
}

#[test]
fn test_so3_isomorphic_to_sl2() {
    let so3_alg = so3();
    let sl2_alg = sl2();
    assert_eq!(so3_alg.dimension, sl2_alg.dimension);
    assert!(so3_alg.is_semisimple());
    assert!(sl2_alg.is_semisimple());
}

// ── Heisenberg tests ──

#[test]
fn test_heisenberg_bracket() {
    let h = heisenberg(1);
    // dim = 3: x0=0, y0=1, z=2
    let r = h.bracket(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
    assert!((r[2] - 1.0).abs() < EPS);
}

#[test]
fn test_heisenberg_jacobi() {
    assert!(heisenberg(1).verify_jacobi());
    assert!(heisenberg(2).verify_jacobi());
}

#[test]
fn test_heisenberg_nilpotent() {
    assert!(heisenberg(1).is_nilpotent(5));
    assert!(heisenberg(2).is_nilpotent(5));
}

#[test]
fn test_heisenberg_not_semisimple() {
    assert!(!heisenberg(1).is_semisimple());
}

#[test]
fn test_heisenberg_center_dim1() {
    let center = heisenberg(1).center();
    assert_eq!(center.len(), 1);
    assert!((center[0][2] - 1.0).abs() < EPS);
}

#[test]
fn test_heisenberg_center_dim_n2() {
    assert_eq!(heisenberg(2).center().len(), 1);
}

#[test]
fn test_heisenberg_solvable() {
    assert!(heisenberg(1).is_solvable(5));
}

#[test]
fn test_heisenberg_not_abelian() {
    assert!(!heisenberg(1).is_abelian());
}

// ── gl(n) tests ──

#[test]
fn test_gl2_bracket() {
    let alg = gl(2);
    // [E00, E01] = E01 (since δ_00·E01 - nothing)
    let r = alg.bracket(&[1.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
    assert!((r[1] - 1.0).abs() < EPS);
}

#[test]
fn test_gl2_jacobi() {
    assert!(gl(2).verify_jacobi());
}

#[test]
fn test_gl2_antisymmetry() {
    assert!(gl(2).verify_antisymmetry());
}

#[test]
fn test_gl2_not_semisimple() {
    assert!(!gl(2).is_semisimple());
}

#[test]
fn test_gl2_center() {
    // Center of gl(2) = scalar matrices, dimension 1
    // E00+E11 is the identity matrix, [E00+E11, anything] = 0
    let center = gl(2).center();
    assert_eq!(center.len(), 1);
    // Center should be spanned by E00 + E11 (the identity matrix)
    let c = &center[0];
    assert!((c[0] - 1.0).abs() < EPS || (c[0] + 1.0).abs() < EPS); // ± identity
    assert!((c[3] - c[0]).abs() < EPS); // same coefficient for E00 and E11
}

// ── Upper triangular tests ──

#[test]
fn test_upper_triangular_3_jacobi() {
    assert!(upper_triangular(3).verify_jacobi());
}

#[test]
fn test_upper_triangular_nilpotent() {
    assert!(upper_triangular(3).is_nilpotent(5));
    assert!(upper_triangular(4).is_nilpotent(5));
}

#[test]
fn test_upper_triangular_2_jacobi() {
    let alg = upper_triangular(2);
    assert!(alg.verify_jacobi());
    assert!(alg.is_abelian());
}

#[test]
fn test_upper_triangular_not_semisimple() {
    assert!(!upper_triangular(3).is_semisimple());
}

// ── Cross product Lie algebra tests ──

#[test]
fn test_cross_product_brackets() {
    let alg = cross_product_lie();
    let r = alg.bracket(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
    assert!((r[2] - 1.0).abs() < EPS);
    let r2 = alg.bracket(&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]);
    assert!((r2[0] - 1.0).abs() < EPS);
    let r3 = alg.bracket(&[0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
    assert!((r3[1] - 1.0).abs() < EPS);
}

#[test]
fn test_cross_product_jacobi() {
    assert!(cross_product_lie().verify_jacobi());
}

#[test]
fn test_cross_product_isomorphic_to_so3() {
    let cp = cross_product_lie();
    let so = so3();
    assert_eq!(cp.dimension, so.dimension);
    assert!(cp.is_semisimple());
    assert!(so.is_semisimple());
}

#[test]
fn test_cross_product_semisimple() {
    assert!(cross_product_lie().is_semisimple());
}

// ── LieRepresentation tests ──

#[test]
fn test_sl2_fundamental_homomorphism() {
    let rep = sl2_fundamental_representation();
    assert!(rep.verify_homomorphism());
}

#[test]
fn test_sl2_fundamental_irreducible() {
    assert!(sl2_fundamental_representation().is_irreducible());
}

#[test]
fn test_sl2_fundamental_character() {
    let rep = sl2_fundamental_representation();
    let ch = rep.character(&[0.0, 1.0, 0.0]);
    assert!(ch.abs() < EPS);
}

#[test]
fn test_sl2_fundamental_rho_e() {
    let rep = sl2_fundamental_representation();
    assert_eq!(rep.rho[0].data[0][1], 1.0);
    assert_eq!(rep.rho[0].data[1][0], 0.0);
}

#[test]
fn test_sl2_fundamental_rho_h() {
    let rep = sl2_fundamental_representation();
    assert_eq!(rep.rho[1].data[0][0], 1.0);
    assert_eq!(rep.rho[1].data[1][1], -1.0);
}

#[test]
fn test_sl2_fundamental_rho_f() {
    let rep = sl2_fundamental_representation();
    assert_eq!(rep.rho[2].data[1][0], 1.0);
    assert_eq!(rep.rho[2].data[0][1], 0.0);
}

#[test]
fn test_direct_sum() {
    let rep = sl2_fundamental_representation();
    let ds = rep.direct_sum(&rep);
    assert_eq!(ds.dimension, 4);
    assert!(ds.verify_homomorphism());
}

#[test]
fn test_direct_sum_not_irreducible() {
    let rep = sl2_fundamental_representation();
    assert!(!rep.direct_sum(&rep).is_irreducible());
}

#[test]
fn test_tensor_product() {
    let rep = sl2_fundamental_representation();
    let tp = rep.tensor_product(&rep);
    assert_eq!(tp.dimension, 4);
    assert!(tp.verify_homomorphism());
}

#[test]
fn test_representation_serde() {
    let rep = sl2_fundamental_representation();
    let json = serde_json::to_string(&rep).unwrap();
    let rep2: LieRepresentation = serde_json::from_str(&json).unwrap();
    assert!(rep2.verify_homomorphism());
}

// ── ExponentialMap tests ──

#[test]
fn test_exp_identity() {
    let alg = sl2();
    let exp = ExponentialMap::exp_of(&alg, &[0.0, 0.0, 0.0], 10);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((exp.data[i][j] - expected).abs() < EPS, "exp(0)[{}][{}] = {} ≠ {}", i, j, exp.data[i][j], expected);
        }
    }
}

#[test]
fn test_bch_zero_y() {
    let alg = sl2();
    let z = ExponentialMap::baker_campbell_hausdorff(&alg, &[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
    assert!((z[0] - 1.0).abs() < EPS);
}

#[test]
fn test_bch_commutative_case() {
    let alg = sl2();
    let x = vec![0.0, 1.0, 0.0];
    let y = vec![0.0, 2.0, 0.0];
    let z = ExponentialMap::baker_campbell_hausdorff(&alg, &x, &y);
    assert!((z[1] - 3.0).abs() < EPS);
}

#[test]
fn test_bch_formula_terms() {
    let alg = sl2();
    let x = vec![1.0, 0.0, 0.0]; // e
    let y = vec![0.0, 0.0, 1.0]; // f
    let z = ExponentialMap::baker_campbell_hausdorff(&alg, &x, &y);
    // BCH: X + Y + ½[X,Y] + 1/12[X,[X,Y]] - 1/12[Y,[X,Y]]
    // [e,f] = h → ½h
    // [e,h] = -2e → 1/12·(-2e)
    // [f,h] = 2f → -1/12·(2f)  Wait: [Y,[X,Y]] = [f,[e,f]] = [f,h] = 2f? 
    // No: [f,h] = -[h,f] = 2f? Let me check: [h,f] = -2f, so [f,h] = 2f
    // Hmm wait, [Y,[X,Y]] = [f, [e,f]] = [f, h]
    // [f, h] = -[h, f] = 2f? No: [h,f] = -2f means [f,h] = 2f.
    // Wait: the bracket [h,f] = -2f means C[1][2][2] = -2, so [f,h] = -[h,f] = 2f.
    // So [f,h] = 2f.
    // [Y,[X,Y]] = [f,h] = -[h,f] = 2f
    // So -1/12·[Y,[X,Y]] = -1/12·2f = -f/6
    //
    // Z = e + f + ½h + 1/12·(-2e) + (-1/12)·(-2f)
    // Wait: [X,[X,Y]] = [e, h] = -2e
    // So 1/12·(-2e) = -e/6
    //
    // [Y,[X,Y]] = [f, h] = -[h,f] = 2f
    // -1/12·(2f) = -f/6
    //
    // Z = e + f + ½h - e/6 - f/6 = 5e/6 + 5f/6 + h/2
    assert!((z[0] - 5.0 / 6.0).abs() < EPS, "z[0] = {} ≠ 5/6", z[0]);
    assert!((z[1] - 0.5).abs() < EPS, "z[1] = {} ≠ 0.5", z[1]);
    assert!((z[2] - 5.0 / 6.0).abs() < EPS, "z[2] = {} ≠ 5/6", z[2]);
}

#[test]
fn test_exp_of_small_element() {
    let alg = sl2();
    let exp = ExponentialMap::exp_of(&alg, &[0.0, 0.1, 0.0], 15);
    let t: f64 = 0.1;
    assert!((exp.data[0][0] - (2.0_f64 * t).exp()).abs() < 1e-6);
    assert!((exp.data[1][1] - 1.0).abs() < EPS);
    assert!((exp.data[2][2] - (-2.0_f64 * t).exp()).abs() < 1e-6);
}

// ── Root System tests ──

#[test]
fn test_sl2_cartan_matrix() {
    let rs = sl2_root_system();
    let cm = rs.cartan_matrix();
    // One simple root, so 1×1 matrix: A = 2(α,α)/(α,α) = 2
    assert!((cm.data[0][0] - 2.0).abs() < EPS);
}

#[test]
fn test_sl2_dynkin_a1() {
    let rs = sl2_root_system();
    let dd = rs.dynkin_diagram();
    assert_eq!(dd.nodes, 1);
    assert!(dd.edges.is_empty());
    assert_eq!(dd.classify(), DynkinType::A(1));
}

#[test]
fn test_sl_n_root_system_a_type() {
    let rs = sl_n_root_system(3);
    let dd = rs.dynkin_diagram();
    match dd.classify() {
        DynkinType::A(n) => assert_eq!(n, 2),
        other => panic!("Expected A₂, got {:?}", other),
    }
}

#[test]
fn test_sl4_root_system_a3() {
    let rs = sl_n_root_system(4);
    match rs.dynkin_diagram().classify() {
        DynkinType::A(n) => assert_eq!(n, 3),
        other => panic!("Expected A₃, got {:?}", other),
    }
}

#[test]
fn test_sl3_positive_roots() {
    let rs = sl_n_root_system(3);
    let pos = rs.positive_roots();
    // sl(3) positive roots: e1-e2, e1-e3, e2-e3
    assert_eq!(pos.len(), 3);
}

#[test]
fn test_sl3_total_roots() {
    assert_eq!(sl_n_root_system(3).roots.len(), 6);
}

#[test]
fn test_sl3_cartan_matrix() {
    let rs = sl_n_root_system(3);
    let cm = rs.cartan_matrix();
    assert!((cm.data[0][0] - 2.0).abs() < EPS);
    assert!((cm.data[0][1] - (-1.0)).abs() < EPS);
    assert!((cm.data[1][0] - (-1.0)).abs() < EPS);
    assert!((cm.data[1][1] - 2.0).abs() < EPS);
}

// ── DynkinDiagram tests ──

#[test]
fn test_dynkin_a1() {
    assert_eq!(DynkinDiagram { nodes: 1, edges: vec![] }.classify(), DynkinType::A(1));
}

#[test]
fn test_dynkin_a2() {
    assert_eq!(DynkinDiagram { nodes: 2, edges: vec![(0, 1, 1)] }.classify(), DynkinType::A(2));
}

#[test]
fn test_dynkin_a3() {
    assert_eq!(DynkinDiagram { nodes: 3, edges: vec![(0, 1, 1), (1, 2, 1)] }.classify(), DynkinType::A(3));
}

#[test]
fn test_dynkin_b2() {
    assert_eq!(DynkinDiagram { nodes: 2, edges: vec![(0, 1, 2)] }.classify(), DynkinType::B(2));
}

#[test]
fn test_dynkin_g2() {
    assert_eq!(DynkinDiagram { nodes: 2, edges: vec![(0, 1, 3)] }.classify(), DynkinType::G2);
}

#[test]
fn test_dynkin_d4() {
    assert_eq!(DynkinDiagram { nodes: 4, edges: vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)] }.classify(), DynkinType::D(4));
}

#[test]
fn test_dynkin_f4() {
    // F₄: chain 4 nodes with multiplicities 1,2,1
    let dd = DynkinDiagram { nodes: 4, edges: vec![(0, 1, 1), (1, 2, 2), (2, 3, 1)] };
    assert_eq!(dd.classify(), DynkinType::F4);
}

#[test]
fn test_dynkin_type_serde() {
    let dt = DynkinType::A(3);
    let json = serde_json::to_string(&dt).unwrap();
    let dt2: DynkinType = serde_json::from_str(&json).unwrap();
    assert_eq!(dt, dt2);
}

#[test]
fn test_dynkin_type_eq() {
    assert_eq!(DynkinType::A(3), DynkinType::A(3));
    assert_ne!(DynkinType::A(3), DynkinType::B(3));
    assert_eq!(DynkinType::G2, DynkinType::G2);
}

// ── Weyl group tests ──

#[test]
fn test_weyl_group_a2() {
    assert_eq!(sl_n_root_system(3).weyl_group_size(), 6);
}

#[test]
fn test_weyl_group_a3() {
    assert_eq!(sl_n_root_system(4).weyl_group_size(), 24);
}

#[test]
fn test_weyl_group_a1() {
    assert_eq!(sl2_root_system().weyl_group_size(), 2);
}

// ── Additional algebra tests ──

#[test]
fn test_abelian_algebra() {
    let alg = LieAlgebra::new("abelian(2)", vec!["a".into(), "b".into()],
        vec![vec![vec![0.0; 2]; 2]; 2]);
    assert!(alg.is_abelian());
    assert!(alg.verify_jacobi());
    assert!(alg.verify_antisymmetry());
    assert!(alg.is_solvable(5));
    assert!(alg.is_nilpotent(5));
    assert!(!alg.is_semisimple());
}

#[test]
fn test_sl2_derived_algebra() {
    assert_eq!(sl2().derived_algebra().len(), 3);
}

#[test]
fn test_heisenberg_derived_algebra() {
    assert_eq!(heisenberg(1).derived_algebra().len(), 1);
}

#[test]
fn test_heisenberg_nilpotent_step2() {
    assert!(heisenberg(1).is_nilpotent(3));
}

#[test]
fn test_upper_triangular_nilpotent_step() {
    assert!(upper_triangular(3).is_nilpotent(5));
}

// ── Serde tests ──

#[test]
fn test_lie_algebra_serde() {
    let alg = sl2();
    let json = serde_json::to_string(&alg).unwrap();
    let alg2: LieAlgebra = serde_json::from_str(&json).unwrap();
    assert_eq!(alg2.name, "sl(2)");
    assert_eq!(alg2.dimension, 3);
    assert!(alg2.verify_jacobi());
}

#[test]
fn test_lie_algebra_serde_heisenberg() {
    let alg = heisenberg(2);
    let json = serde_json::to_string(&alg).unwrap();
    let alg2: LieAlgebra = serde_json::from_str(&json).unwrap();
    assert!(alg2.verify_jacobi());
    assert!(alg2.is_nilpotent(5));
}

#[test]
fn test_root_system_serde() {
    let rs = sl_n_root_system(3);
    let json = serde_json::to_string(&rs).unwrap();
    let rs2: RootSystem = serde_json::from_str(&json).unwrap();
    assert_eq!(rs2.roots.len(), 6);
}

#[test]
fn test_dynkin_diagram_serde() {
    let dd = DynkinDiagram { nodes: 3, edges: vec![(0, 1, 1), (1, 2, 1)] };
    let json = serde_json::to_string(&dd).unwrap();
    let dd2: DynkinDiagram = serde_json::from_str(&json).unwrap();
    assert_eq!(dd2.classify(), DynkinType::A(3));
}

// ── Integration tests ──

#[test]
fn test_sl2_exp_e_plus_exp_f_approx_bch() {
    let alg = sl2();
    let e = vec![1.0, 0.0, 0.0];
    let f = vec![0.0, 0.0, 1.0];
    let z = ExponentialMap::baker_campbell_hausdorff(&alg, &e, &f);
    assert!(z.iter().any(|&v| v.abs() > EPS));
}

#[test]
fn test_killing_form_sl2_nondegenerate() {
    let det = sl2().killing_form().determinant();
    assert!(det.abs() > EPS, "Killing form determinant = {}", det);
}

#[test]
fn test_killing_form_heisenberg_degenerate() {
    let det = heisenberg(1).killing_form().determinant();
    assert!(det.abs() < EPS);
}

#[test]
fn test_so3_killing_form() {
    let det = so3().killing_form().determinant();
    assert!(det.abs() > EPS);
}

#[test]
fn test_sl_n_dynkin_an_minus_1() {
    for n in 2..=5 {
        let rs = sl_n_root_system(n);
        match rs.dynkin_diagram().classify() {
            DynkinType::A(m) => assert_eq!(m, n - 1),
            other => panic!("sl({}) → {:?}, expected A({})", n, other, n - 1),
        }
    }
}

#[test]
fn test_adjoint_representation_is_representation() {
    let alg = sl2();
    let n = alg.dimension;
    let rho: Vec<DenseMatrix> = (0..n)
        .map(|i| alg.adjoint_matrix(&super::unit_vec(i, n)))
        .collect();
    let adj_rep = LieRepresentation::new(alg, n, rho);
    assert!(adj_rep.verify_homomorphism());
}

