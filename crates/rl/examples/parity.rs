//! Debug tool: first decision where the trainer's env and `RlStrategy` disagree.
//! `cargo run --release -p rl --example parity -- <model dir> <session dir>`
use market::{Episode, Sim, SimConfig};
use strategy::features::position_features;
use strategy::net::{greedy, masked_softmax};
use strategy::rl::Model;
use strategy::{choice, Action, Strategy};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (model_dir, ep_dir) = (&args[1], &args[2]);
    let ep = Episode::load(ep_dir.as_ref())?;
    let model = Model::load(model_dir.as_ref())?;
    let limits = model.meta.limits;
    let prep = rl::Prepared::new(ep.clone(), limits, model.meta.trunk.every_s);
    let enc = rl::Encoded::new(rl::Prepared::new(ep.clone(), limits, model.meta.trunk.every_s), &model);
    let cfg = SimConfig { latency_s: (1, 1), limits, ..Default::default() };

    let mut strat = strategy::RlStrategy::load(model_dir.as_ref())?;
    strat.begin_window(&ep.window);
    let mut sim = Sim::new(&ep, cfg, 0);
    let w = enc.width;
    let mut k = 0;
    for (i, s) in ep.snapshots.iter().enumerate() {
        let d = strat.on_tick(s, &sim.pos);
        if k < prep.points.len() && prep.points[k].0 == i {
            let (l, _) = model.head_one(&enc.market[k * w..(k + 1) * w], &position_features(&sim.pos, s));
            let mask = limits.mask(s, &sim.pos);
            let p = masked_softmax(&l, &mask);
            let (a, frac) = choice(greedy(&p, &mask));
            let size = limits.order_size(a, frac, s, &sim.pos);
            if (a, size) != (d.action, d.size) && !(size == 0 && d.action == Action::Nothing) {
                println!("diverge at idx {i} (k={k}): env {a:?} x{size} strat {:?} x{}\n env p {p:?}\n strat p {:?}", d.action, d.size, d.probs);
                return Ok(());
            }
            k += 1;
        } else if d.action != Action::Nothing {
            println!("strategy acted at idx {i} which is not a prepared point (k={k}, next point {:?})", prep.points.get(k).map(|p| p.0));
            return Ok(());
        }
        if d.action != Action::Nothing {
            sim.act(i, d.action, d.size);
        }
    }
    println!("no divergence over {} points; strategy-path pnl {:.4}", k, sim.settle());
    let pcfg = rl::PpoConfig { limits, eval_sim: cfg, ..Default::default() };
    let one = rl::ppo::evaluate(&model.head, std::slice::from_ref(&enc), &pcfg);
    let r = market::run_strategy(&mut strategy::RlStrategy::load(model_dir.as_ref())?, &ep, cfg, 0);
    println!("evaluate() pnl {:.4}, run_strategy pnl {:.4}", one.total_pnl, r.pnl);
    Ok(())
}
