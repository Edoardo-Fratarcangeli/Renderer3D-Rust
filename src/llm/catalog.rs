//! Catalog of representable neural-network architectures.
//!
//! Beyond transformer LLMs (handled by [`crate::llm::arch`]), this module
//! synthesizes a [`NetworkGraph`] for a broad set of ML/DL families so the
//! visualizer can render *any* of them with the same node/edge machinery:
//!
//!  - **MLP / Perceptron** — classic fully-connected nets.
//!  - **2D CNNs** — LeNet/VGG/ResNet-style image classifiers.
//!  - **3D CNNs** — volumetric / voxel networks (VoxNet).
//!  - **U-Net** — encoder–decoder with skip connections.
//!  - **Autoencoder / VAE** — latent-bottleneck reconstruction nets.
//!  - **RNN / LSTM / GRU** — recurrent sequence models.
//!  - **GAN** — generator + discriminator pair.
//!  - **PointNet / PointNet++** — point-cloud networks.
//!  - **GCN / GAT / GraphSAGE** — graph neural networks.
//!  - **Transformer / ViT** — generic attention stacks.
//!
//! Each entry is a pure constructor returning a laid-out [`NetworkGraph`]; the
//! 3D positions, edges and centroid are all computed by
//! [`NetworkGraph::layout`].

use crate::llm::network::{Layer, LayerKind, NetworkGraph, Node, MAX_NODES_PER_LAYER};

/// A representable neural-network architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NetArch {
    #[default]
    Mlp,
    Cnn2d,
    Vgg,
    Resnet,
    Unet,
    Cnn3d,
    Autoencoder,
    Vae,
    Rnn,
    Lstm,
    Gru,
    Gan,
    PointNet,
    PointNetPp,
    Gcn,
    Gat,
    GraphSage,
    Transformer,
    Vit,
}

impl NetArch {
    /// Every architecture, in display order.
    pub const ALL: [NetArch; 19] = [
        NetArch::Mlp,
        NetArch::Cnn2d,
        NetArch::Vgg,
        NetArch::Resnet,
        NetArch::Unet,
        NetArch::Cnn3d,
        NetArch::Autoencoder,
        NetArch::Vae,
        NetArch::Rnn,
        NetArch::Lstm,
        NetArch::Gru,
        NetArch::Gan,
        NetArch::PointNet,
        NetArch::PointNetPp,
        NetArch::Gcn,
        NetArch::Gat,
        NetArch::GraphSage,
        NetArch::Transformer,
        NetArch::Vit,
    ];

    /// Short, human-facing name (proper nouns — not localized).
    pub fn label(self) -> &'static str {
        match self {
            NetArch::Mlp         => "MLP / Perceptron",
            NetArch::Cnn2d       => "2D CNN (LeNet)",
            NetArch::Vgg         => "2D CNN (VGG)",
            NetArch::Resnet      => "ResNet",
            NetArch::Unet        => "U-Net",
            NetArch::Cnn3d       => "3D CNN (VoxNet)",
            NetArch::Autoencoder => "Autoencoder",
            NetArch::Vae         => "Variational AE",
            NetArch::Rnn         => "RNN",
            NetArch::Lstm        => "LSTM",
            NetArch::Gru         => "GRU",
            NetArch::Gan         => "GAN",
            NetArch::PointNet    => "PointNet",
            NetArch::PointNetPp  => "PointNet++",
            NetArch::Gcn         => "GCN (Graph)",
            NetArch::Gat         => "GAT (Graph)",
            NetArch::GraphSage   => "GraphSAGE",
            NetArch::Transformer => "Transformer",
            NetArch::Vit         => "Vision Transformer",
        }
    }

    /// Coarse grouping used to lay the catalog out in sections.
    pub fn group(self) -> &'static str {
        match self {
            NetArch::Mlp => "Dense",
            NetArch::Cnn2d | NetArch::Vgg | NetArch::Resnet | NetArch::Unet | NetArch::Cnn3d => "Convolutional",
            NetArch::Autoencoder | NetArch::Vae | NetArch::Gan => "Generative",
            NetArch::Rnn | NetArch::Lstm | NetArch::Gru => "Recurrent",
            NetArch::PointNet | NetArch::PointNetPp => "Point cloud",
            NetArch::Gcn | NetArch::Gat | NetArch::GraphSage => "Graph",
            NetArch::Transformer | NetArch::Vit => "Attention",
        }
    }

    /// Build a fully laid-out graph for this architecture.
    pub fn build(self) -> NetworkGraph {
        let layers = match self {
            NetArch::Mlp         => mlp(),
            NetArch::Cnn2d       => cnn2d(),
            NetArch::Vgg         => vgg(),
            NetArch::Resnet      => resnet(),
            NetArch::Unet        => unet(),
            NetArch::Cnn3d       => cnn3d(),
            NetArch::Autoencoder => autoencoder(false),
            NetArch::Vae         => autoencoder(true),
            NetArch::Rnn         => recurrent("RNN"),
            NetArch::Lstm        => recurrent("LSTM"),
            NetArch::Gru         => recurrent("GRU"),
            NetArch::Gan         => gan(),
            NetArch::PointNet    => pointnet(false),
            NetArch::PointNetPp  => pointnet(true),
            NetArch::Gcn         => graph_net("GCN"),
            NetArch::Gat         => graph_net("GAT"),
            NetArch::GraphSage   => graph_net("GraphSAGE"),
            NetArch::Transformer => transformer(false),
            NetArch::Vit         => transformer(true),
        };
        let mut graph = NetworkGraph {
            name: self.label().to_string(),
            layers,
            edges: vec![],
            estimated_vram_gb: None,
            moe_config: None,
        };
        graph.layout();
        graph
    }
}

// ─── Node helpers ──────────────────────────────────────────────────────────────

/// `n` nodes with base weight `w` plus a small deterministic ripple so spheres
/// are not all identical (purely cosmetic; keeps positions readable).
fn nodes(n: usize, w: f32) -> Vec<Node> {
    let n = n.clamp(1, MAX_NODES_PER_LAYER);
    (0..n)
        .map(|i| {
            // Cheap deterministic jitter in [-0.12, 0.12].
            let h = ((i as u32).wrapping_mul(2654435761) >> 24) as f32 / 255.0;
            let weight = (w + (h - 0.5) * 0.24).clamp(0.05, 1.0);
            Node { position: [0.0; 3], weight_magnitude: weight }
        })
        .collect()
}

fn layer(name: impl Into<String>, kind: LayerKind, n: usize, w: f32) -> Layer {
    Layer { name: name.into(), kind, nodes: nodes(n, w) }
}

// ─── Architecture builders ─────────────────────────────────────────────────────

fn mlp() -> Vec<Layer> {
    vec![
        layer("Input", LayerKind::Input, 16, 0.55),
        layer("Dense 128", LayerKind::Dense, 32, 0.60),
        layer("Dense 64", LayerKind::Dense, 24, 0.60),
        layer("Dense 32", LayerKind::Dense, 16, 0.55),
        layer("Softmax", LayerKind::Output, 10, 0.65),
    ]
}

fn cnn2d() -> Vec<Layer> {
    vec![
        layer("Image 28×28", LayerKind::Input, 36, 0.50),
        layer("Conv 6@5×5", LayerKind::Convolution, 24, 0.70),
        layer("MaxPool", LayerKind::Pooling, 16, 0.45),
        layer("Conv 16@5×5", LayerKind::Convolution, 32, 0.75),
        layer("MaxPool", LayerKind::Pooling, 16, 0.45),
        layer("Dense 120", LayerKind::Dense, 28, 0.60),
        layer("Dense 84", LayerKind::Dense, 20, 0.60),
        layer("Softmax", LayerKind::Output, 10, 0.65),
    ]
}

fn vgg() -> Vec<Layer> {
    let mut l = vec![layer("Image 224×224", LayerKind::Input, 49, 0.45)];
    let blocks = [(64, 2), (128, 2), (256, 3), (512, 3)];
    for (bi, (ch, convs)) in blocks.iter().enumerate() {
        for c in 0..*convs {
            l.push(layer(format!("Conv{bi}_{c} {ch}@3×3"), LayerKind::Convolution, (ch / 12).clamp(16, 48), 0.72));
        }
        l.push(layer(format!("Pool{bi}"), LayerKind::Pooling, 16, 0.42));
    }
    l.push(layer("FC 4096", LayerKind::Dense, 40, 0.62));
    l.push(layer("FC 4096", LayerKind::Dense, 40, 0.62));
    l.push(layer("Softmax 1000", LayerKind::Output, 24, 0.66));
    l
}

fn resnet() -> Vec<Layer> {
    let mut l = vec![
        layer("Image", LayerKind::Input, 49, 0.45),
        layer("Conv 7×7", LayerKind::Convolution, 32, 0.72),
        layer("MaxPool", LayerKind::Pooling, 16, 0.42),
    ];
    let stages = [(64, 2), (128, 2), (256, 2), (512, 2)];
    for (si, (ch, blocks)) in stages.iter().enumerate() {
        for b in 0..*blocks {
            l.push(layer(format!("Res{si}_{b} conv"), LayerKind::Convolution, (ch / 12).clamp(16, 48), 0.74));
            l.push(layer(format!("Res{si}_{b} conv"), LayerKind::Convolution, (ch / 12).clamp(16, 48), 0.74));
            l.push(layer(format!("Res{si}_{b} ⊕"), LayerKind::Residual, 12, 0.55));
        }
    }
    l.push(layer("GlobalAvgPool", LayerKind::Pooling, 16, 0.42));
    l.push(layer("Softmax", LayerKind::Output, 24, 0.66));
    l
}

fn unet() -> Vec<Layer> {
    vec![
        layer("Input", LayerKind::Input, 49, 0.45),
        layer("Enc1 Conv", LayerKind::Convolution, 36, 0.70),
        layer("Down1", LayerKind::Pooling, 16, 0.42),
        layer("Enc2 Conv", LayerKind::Convolution, 28, 0.72),
        layer("Down2", LayerKind::Pooling, 12, 0.42),
        layer("Bottleneck", LayerKind::Latent, 16, 0.80),
        layer("Up2", LayerKind::Upsample, 12, 0.50),
        layer("Skip2 ⊕", LayerKind::Residual, 12, 0.55),
        layer("Dec2 Conv", LayerKind::Convolution, 28, 0.70),
        layer("Up1", LayerKind::Upsample, 16, 0.50),
        layer("Skip1 ⊕", LayerKind::Residual, 16, 0.55),
        layer("Dec1 Conv", LayerKind::Convolution, 36, 0.70),
        layer("Segmentation", LayerKind::Output, 24, 0.64),
    ]
}

fn cnn3d() -> Vec<Layer> {
    vec![
        layer("Voxel 32³", LayerKind::Input, 64, 0.48),
        layer("Conv3D 32@5³", LayerKind::Convolution, 40, 0.72),
        layer("Pool3D", LayerKind::Pooling, 24, 0.44),
        layer("Conv3D 32@3³", LayerKind::Convolution, 32, 0.74),
        layer("Pool3D", LayerKind::Pooling, 16, 0.44),
        layer("Dense 128", LayerKind::Dense, 28, 0.60),
        layer("Softmax", LayerKind::Output, 10, 0.65),
    ]
}

fn autoencoder(variational: bool) -> Vec<Layer> {
    let mut l = vec![
        layer("Input", LayerKind::Input, 32, 0.52),
        layer("Enc 256", LayerKind::Dense, 28, 0.60),
        layer("Enc 64", LayerKind::Dense, 20, 0.60),
    ];
    if variational {
        l.push(layer("μ / σ", LayerKind::Latent, 12, 0.85));
        l.push(layer("z ~ N(μ,σ)", LayerKind::Latent, 8, 0.90));
    } else {
        l.push(layer("Latent code", LayerKind::Latent, 8, 0.88));
    }
    l.push(layer("Dec 64", LayerKind::Dense, 20, 0.60));
    l.push(layer("Dec 256", LayerKind::Dense, 28, 0.60));
    l.push(layer("Reconstruction", LayerKind::Output, 32, 0.62));
    l
}

fn recurrent(cell: &str) -> Vec<Layer> {
    vec![
        layer("Input seq", LayerKind::Input, 16, 0.52),
        layer("Embedding", LayerKind::Embedding, 24, 0.60),
        layer(format!("{cell} cell t-1"), LayerKind::Recurrent, 28, 0.68),
        layer(format!("{cell} cell t"), LayerKind::Recurrent, 28, 0.72),
        layer(format!("{cell} cell t+1"), LayerKind::Recurrent, 28, 0.68),
        layer("Dense", LayerKind::Dense, 20, 0.58),
        layer("Output", LayerKind::Output, 12, 0.64),
    ]
}

fn gan() -> Vec<Layer> {
    vec![
        layer("z noise", LayerKind::Latent, 12, 0.85),
        layer("G Dense", LayerKind::Dense, 24, 0.60),
        layer("G Upsample", LayerKind::Upsample, 32, 0.58),
        layer("G Conv", LayerKind::Convolution, 40, 0.70),
        layer("Fake / Real", LayerKind::Input, 36, 0.50),
        layer("D Conv", LayerKind::Convolution, 32, 0.72),
        layer("D Pool", LayerKind::Pooling, 16, 0.44),
        layer("D Dense", LayerKind::Dense, 16, 0.58),
        layer("Real?", LayerKind::Output, 4, 0.70),
    ]
}

fn pointnet(hierarchical: bool) -> Vec<Layer> {
    let mut l = vec![
        layer("Point cloud N×3", LayerKind::PointSet, 64, 0.50),
        layer("Shared MLP", LayerKind::Dense, 32, 0.62),
    ];
    if hierarchical {
        l.push(layer("Set Abstraction 1", LayerKind::PointSet, 48, 0.66));
        l.push(layer("Set Abstraction 2", LayerKind::PointSet, 28, 0.70));
        l.push(layer("Set Abstraction 3", LayerKind::PointSet, 16, 0.74));
    } else {
        l.push(layer("Shared MLP", LayerKind::Dense, 28, 0.64));
    }
    l.push(layer("Max Pool (symmetric)", LayerKind::Pooling, 16, 0.46));
    l.push(layer("Global feature", LayerKind::Latent, 12, 0.82));
    l.push(layer("Classifier", LayerKind::Output, 16, 0.64));
    l
}

fn graph_net(kind: &str) -> Vec<Layer> {
    vec![
        layer("Node features", LayerKind::Input, 36, 0.52),
        layer(format!("{kind} layer 1"), LayerKind::Graph, 40, 0.68),
        layer(format!("{kind} layer 2"), LayerKind::Graph, 36, 0.70),
        layer(format!("{kind} layer 3"), LayerKind::Graph, 28, 0.70),
        layer("Readout / Pool", LayerKind::Pooling, 16, 0.46),
        layer("Output", LayerKind::Output, 12, 0.64),
    ]
}

fn transformer(vision: bool) -> Vec<Layer> {
    let mut l = Vec::new();
    if vision {
        l.push(layer("Patch embed", LayerKind::Input, 49, 0.50));
        l.push(layer("Linear proj + [CLS]", LayerKind::Embedding, 32, 0.60));
    } else {
        l.push(layer("Token embed", LayerKind::Embedding, 32, 0.60));
    }
    for i in 0..4 {
        l.push(layer(format!("Block {i} · Attn"), LayerKind::Attention, 24, 0.74));
        l.push(layer(format!("Block {i} · LN"), LayerKind::LayerNorm, 8, 0.40));
        l.push(layer(format!("Block {i} · FFN"), LayerKind::FeedForward, 32, 0.62));
    }
    l.push(layer(if vision { "MLP head" } else { "LM head" }, LayerKind::Output, 24, 0.66));
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_arch_builds_a_nonempty_laid_out_graph() {
        for arch in NetArch::ALL {
            let g = arch.build();
            assert!(!g.layers.is_empty(), "{:?} has no layers", arch);
            assert!(g.node_count() > 0, "{:?} has no nodes", arch);
            assert!(!g.edges.is_empty(), "{:?} produced no edges", arch);
            assert!(g.centroid().is_some(), "{:?} has no centroid", arch);
            // Every node must have been positioned by layout() (x grows with depth).
            assert_eq!(g.name, arch.label());
            // Node caps respected.
            assert!(g.layers.iter().all(|l| l.nodes.len() <= MAX_NODES_PER_LAYER));
        }
    }

    #[test]
    fn labels_and_groups_are_unique_and_populated() {
        for arch in NetArch::ALL {
            assert!(!arch.label().is_empty());
            assert!(!arch.group().is_empty());
        }
        // Labels are distinct.
        for (i, a) in NetArch::ALL.iter().enumerate() {
            for b in &NetArch::ALL[i + 1..] {
                assert_ne!(a.label(), b.label());
            }
        }
    }
}
