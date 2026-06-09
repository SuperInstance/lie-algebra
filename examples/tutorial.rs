//! # Lie Algebra Tutorial
//!
//! Progressive lessons covering Lie brackets, structure constants, classical
//! algebras, Killing forms, representations, root systems, and Dynkin diagrams.
//!
//! Run with: `cargo run --example tutorial`

use lie_algebra::{
    cross_product_lie, gl, heisenberg, sl2, sl2_fundamental_representation,
    sl2_root_system, sl_n_root_system, so3, upper_triangular, DenseMatrix,
    DynkinDiagram, DynkinType, ExponentialMap, LieAlgebra, LieRepresentation,
    RootSystem,
};

// ── Lesson 1: What is a Lie Algebra? ─────────────────────────────────────────

fn lesson_1_what_is_lie_algebra() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 1: What is a Lie Algebra?");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A Lie algebra is a vector space with a bilinear bracket [·,·] satisfying:");
    println!("  1. Antisymmetry: [X,Y] = −[Y,X]");
    println!("  2. Jacobi identity: [X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0");
    println!();

    // Create a custom Lie algebra: ℝ³ with cross product
    let alg = cross_product_lie();
    println!("  Example: ℝ³ with cross product as bracket");
    println!("    Name: {}", alg.name);
    println!("    Basis: {:?}", alg.basis);
    println!("    Dimension: {}", alg.dimension);
    println!();

    // Compute [e₁, e₂] = e₃
    let e1 = vec![1.0, 0.0, 0.0];
    let e2 = vec![0.0, 1.0, 0.0];
    let bracket = alg.bracket(&e1, &e2);
    println!("  [e₁, e₂] = {:?}", bracket);
    println!("  [e₂, e₃] = {:?}", alg.bracket(&e2, &[0.0, 0.0, 1.0]));
    println!("  [e₃, e₁] = {:?}", alg.bracket(&[0.0, 0.0, 1.0], &e1));
    println!();

    println!("  Antisymmetric? {}", alg.verify_antisymmetry());
    println!("  Jacobi identity? {}", alg.verify_jacobi());
    println!();
}

// ── Lesson 2: Structure Constants ────────────────────────────────────────────

fn lesson_2_structure_constants() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 2: Structure Constants");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("Structure constants C[i][j][k] define the bracket:");
    println!("  [eᵢ, eⱼ] = Σₖ C[i][j][k] · eₖ");
    println!();

    let alg = sl2();
    println!("  sl(2,ℝ) — the special linear Lie algebra");
    println!("    Basis: {:?}  (e=raising, h=Cartan, f=lowering)", alg.basis);
    println!();

    // Display the nonzero brackets
    let brackets = [
        ([0, 2], "e, f"), // [e,f]
        ([1, 0], "h, e"), // [h,e]
        ([1, 2], "h, f"), // [h,f]
    ];
    println!("    Nonzero brackets:");
    for ([i, j], label) in &brackets {
        let a = lie_algebra::unit_vec(*i, 3);
        let b = lie_algebra::unit_vec(*j, 3);
        let br = alg.bracket(&a, &b);
        let terms: Vec<String> = br.iter().enumerate()
            .filter(|(_, &v)| v.abs() > 1e-10)
            .map(|(k, &v)| format!("{:.0}·{}", v, alg.basis[k]))
            .collect();
        println!("      [{}] = {}", label, terms.join(" + "));
    }
    println!();
}

// ── Lesson 3: Classical Lie Algebras ─────────────────────────────────────────

fn lesson_3_classical_algebras() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 3: Classical Lie Algebras");
    println!("═══════════════════════════════════════════════════");
    println!();

    // sl(2)
    let sl2_alg = sl2();
    println!("  sl(2): dim={}, abelian={}, semisimple={}",
             sl2_alg.dimension, sl2_alg.is_abelian(), sl2_alg.is_semisimple());
    println!("    Jacobi: {}", sl2_alg.verify_jacobi());
    println!();

    // so(3)
    let so3_alg = so3();
    println!("  so(3): dim={}, semisimple={}, isomorphic to sl(2)? {}",
             so3_alg.dimension, so3_alg.is_semisimple(), so3_alg.dimension == sl2_alg.dimension);
    println!("    Basis: {:?}", so3_alg.basis);
    println!("    [L1, L2] = {:?}", so3_alg.bracket(&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]));
    println!();

    // Heisenberg
    let h1 = heisenberg(1);
    println!("  Heisenberg(1): dim={}, abelian={}, nilpotent={}",
             h1.dimension, h1.is_abelian(), h1.is_nilpotent(5));
    println!("    Basis: {:?}", h1.basis);
    println!("    Center dimension: {}", h1.center().len());
    println!("    Solvable: {}", h1.is_solvable(5));
    println!();

    // gl(2)
    let gl2 = gl(2);
    println!("  gl(2): dim={}, semisimple={}",
             gl2.dimension, gl2.is_semisimple());
    println!("    Center dimension: {} (scalar matrices)", gl2.center().len());
    println!();

    // Upper triangular
    let ut3 = upper_triangular(3);
    println!("  upper_triangular(3): dim={}, nilpotent={}",
             ut3.dimension, ut3.is_nilpotent(5));
    println!();
}

// ── Lesson 4: The Killing Form ───────────────────────────────────────────────

fn lesson_4_killing_form() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 4: The Killing Form & Cartan's Criterion");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("The Killing form κ(x,y) = tr(ad(x)∘ad(y)) is a symmetric bilinear form.");
    println!("Cartan's criterion: the algebra is semisimple ⟺ κ is non-degenerate.");
    println!();

    // sl(2) Killing form
    let sl2_alg = sl2();
    let kf = sl2_alg.killing_form();
    println!("  sl(2) Killing form (basis: e, h, f):");
    for row in &kf.data {
        println!("    {:?}", row);
    }
    println!("    det(κ) = {:.1} (non-degenerate → semisimple)", kf.determinant());
    println!();

    // Heisenberg Killing form (degenerate)
    let h1 = heisenberg(1);
    let kf_h = h1.killing_form();
    println!("  Heisenberg(1) Killing form:");
    for row in &kf_h.data {
        println!("    {:?}", row);
    }
    println!("    det(κ) = {:.1} (degenerate → NOT semisimple)", kf_h.determinant());
    println!();

    // so(3) Killing form
    let so3_alg = so3();
    let kf_so3 = so3_alg.killing_form();
    println!("  so(3) Killing form:");
    println!("    det(κ) = {:.1} (non-degenerate → semisimple)", kf_so3.determinant());
    println!();
}

// ── Lesson 5: Center, Derived Algebra & Structural Properties ────────────────

fn lesson_5_structural() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 5: Center, Derived Algebra & Structural Properties");
    println!("═══════════════════════════════════════════════════");
    println!();

    let algebras: Vec<(&str, LieAlgebra)> = vec![
        ("sl(2)", sl2()),
        ("so(3)", so3()),
        ("heisenberg(1)", heisenberg(1)),
        ("gl(2)", gl(2)),
        ("upper_tri(3)", upper_triangular(3)),
    ];

    println!("  {:15} {:>5} {:>5} {:>6} {:>7} {:>8} {:>9} {:>10}",
             "Algebra", "dim", "ctr", "abel", "semis", "solvable", "nilpot", "derived");
    println!("  {}", "─".repeat(80));

    for (name, alg) in &algebras {
        println!("  {:15} {:>5} {:>5} {:>6} {:>7} {:>8} {:>9} {:>10}",
                 name,
                 alg.dimension,
                 alg.center().len(),
                 alg.is_abelian(),
                 alg.is_semisimple(),
                 alg.is_solvable(10),
                 alg.is_nilpotent(10),
                 alg.derived_algebra().len());
    }
    println!();
}

// ── Lesson 6: Lie Representations ────────────────────────────────────────────

fn lesson_6_representations() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 6: Lie Representations");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("A representation ρ maps Lie algebra elements to matrices preserving brackets:");
    println!("  ρ([X,Y]) = ρ(X)ρ(Y) − ρ(Y)ρ(X) = [ρ(X), ρ(Y)]");
    println!();

    // sl(2) fundamental representation (2-dimensional)
    let rep = sl2_fundamental_representation();
    println!("  sl(2) fundamental (2-dim) representation:");
    println!("    Basis images:");
    for (i, name) in ["e", "h", "f"].iter().enumerate() {
        println!("      ρ({}) = {:?}", name, rep.rho[i].data);
    }
    println!();
    println!("    Homomorphism verified? {}", rep.verify_homomorphism());
    println!("    Irreducible? {}", rep.is_irreducible());
    println!();

    // Character
    let h_elem = [0.0, 1.0, 0.0]; // h element
    let ch = rep.character(&h_elem);
    println!("    Character of h: tr(ρ(h)) = {:.0}  (= 1 + (-1) = 0)", ch);
    println!();

    // Direct sum
    let ds = rep.direct_sum(&rep);
    println!("    Direct sum V ⊕ V: dimension={}, irreducible? {}", ds.dimension, ds.is_irreducible());
    println!("    Homomorphism verified? {}", ds.verify_homomorphism());
    println!();

    // Tensor product
    let tp = rep.tensor_product(&rep);
    println!("    Tensor product V ⊗ V: dimension={}", tp.dimension);
    println!("    Homomorphism verified? {}", tp.verify_homomorphism());
    println!();
}

// ── Lesson 7: Root Systems & Dynkin Diagrams ─────────────────────────────────

fn lesson_7_root_systems() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 7: Root Systems & Dynkin Diagrams");
    println!("═══════════════════════════════════════════════════");
    println!();

    // sl(2) root system
    let rs2 = sl2_root_system();
    println!("  sl(2) root system (type A₁):");
    println!("    Roots: {:?}", rs2.roots);
    println!("    Simple roots: {:?}", rs2.simple_roots);
    println!("    Cartan matrix: {:?}", rs2.cartan_matrix().data);
    println!("    Dynkin type: {:?}", rs2.dynkin_diagram().classify());
    println!("    Weyl group size: |W| = {}", rs2.weyl_group_size());
    println!();

    // sl(3) root system
    let rs3 = sl_n_root_system(3);
    println!("  sl(3) root system (type A₂):");
    println!("    Total roots: {} (= 2 × n(n-1) for sl(n))", rs3.roots.len());
    println!("    Positive roots: {:?}", rs3.positive_roots());
    println!("    Simple roots: {:?}", rs3.simple_roots);
    println!("    Cartan matrix:");
    for row in &rs3.cartan_matrix().data {
        println!("      {:?}", row);
    }
    println!("    Dynkin type: {:?}", rs3.dynkin_diagram().classify());
    println!("    Weyl group size: |W| = {} (= 3! = S₃)", rs3.weyl_group_size());
    println!();

    // sl(4) root system
    let rs4 = sl_n_root_system(4);
    println!("  sl(4) root system (type A₃):");
    println!("    Total roots: {}", rs4.roots.len());
    println!("    Dynkin type: {:?}", rs4.dynkin_diagram().classify());
    println!("    Weyl group size: |W| = {} (= 4!)", rs4.weyl_group_size());
    println!();
}

// ── Lesson 8: Exponential Map & BCH Formula ───────────────────────────────────

fn lesson_8_exponential_bch() {
    println!("═══════════════════════════════════════════════════");
    println!("  Lesson 8: Exponential Map & BCH Formula");
    println!("═══════════════════════════════════════════════════");
    println!();
    println!("The exponential map exp: 𝔤 → G maps Lie algebra elements to the Lie group.");
    println!("The BCH formula computes exp(X)·exp(Y) = exp(Z) in the Lie algebra.");
    println!();

    let alg = sl2();

    // exp(0) = identity
    let exp_0 = ExponentialMap::exp_of(&alg, &[0.0, 0.0, 0.0], 10);
    println!("  exp(0) =");
    for row in &exp_0.data {
        println!("    {:?}", row);
    }
    println!();

    // exp(small · h)
    let t = 0.1;
    let exp_h = ExponentialMap::exp_of(&alg, &[0.0, t, 0.0], 15);
    println!("  exp({:.1}·h) =", t);
    for row in &exp_h.data {
        println!("    [{:.6}, {:.6}, {:.6}]", row[0], row[1], row[2]);
    }
    println!("    (diagonal: e^(2t), 1, e^(-2t) = {:.6}, 1, {:.6})",
             (2.0 * t).exp(), (-2.0 * t).exp());
    println!();

    // BCH formula
    let x = vec![1.0, 0.0, 0.0]; // e
    let y = vec![0.0, 0.0, 1.0]; // f
    let z = ExponentialMap::baker_campbell_hausdorff(&alg, &x, &y);
    println!("  BCH(exp(e) · exp(f)) to order 3:");
    println!("    Z = e + f + ½[e,f] + 1/12[e,[e,f]] − 1/12[f,[e,f]]");
    let terms: Vec<String> = z.iter().enumerate()
        .filter(|(_, &v)| v.abs() > 1e-10)
        .map(|(k, &v)| format!("{:.4}·{}", v, alg.basis[k]))
        .collect();
    println!("    Z = {}", terms.join(" + "));
    println!();

    // BCH with commuting elements
    let h1 = vec![0.0, 1.0, 0.0];
    let h2 = vec![0.0, 2.0, 0.0];
    let z_comm = ExponentialMap::baker_campbell_hausdorff(&alg, &h1, &h2);
    println!("  BCH(exp(h) · exp(2h)) = exp(3h) since [h,h]=0:");
    println!("    Z = {:?}  (exactly h₁ + h₂)", z_comm);
    println!();

    // Dynkin diagram classification showcase
    println!("  Dynkin diagram classification:");
    let diagrams = vec![
        ("A₁", DynkinDiagram { nodes: 1, edges: vec![] }),
        ("A₃", DynkinDiagram { nodes: 3, edges: vec![(0, 1, 1), (1, 2, 1)] }),
        ("B₂", DynkinDiagram { nodes: 2, edges: vec![(0, 1, 2)] }),
        ("D₄", DynkinDiagram { nodes: 4, edges: vec![(0, 1, 1), (1, 2, 1), (1, 3, 1)] }),
        ("G₂", DynkinDiagram { nodes: 2, edges: vec![(0, 1, 3)] }),
        ("F₄", DynkinDiagram { nodes: 4, edges: vec![(0, 1, 1), (1, 2, 2), (2, 3, 1)] }),
    ];
    for (name, dd) in &diagrams {
        println!("    {} → {:?}", name, dd.classify());
    }
    println!();
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   Lie Algebra Tutorial                           ║");
    println!("║   Brackets, Representations, Root Systems        ║");
    println!("╚═══════════════════════════════════════════════════╝");
    println!();

    lesson_1_what_is_lie_algebra();
    lesson_2_structure_constants();
    lesson_3_classical_algebras();
    lesson_4_killing_form();
    lesson_5_structural();
    lesson_6_representations();
    lesson_7_root_systems();
    lesson_8_exponential_bch();

    println!("═══════════════════════════════════════════════════");
    println!("  Tutorial complete! Key takeaways:");
    println!("    • Lie brackets encode infinitesimal symmetries");
    println!("    • Structure constants define any finite-dimensional Lie algebra");
    println!("    • Killing form non-degeneracy ⟺ semisimplicity (Cartan)");
    println!("    • Root systems & Dynkin diagrams classify simple Lie algebras");
    println!("═══════════════════════════════════════════════════");
}
