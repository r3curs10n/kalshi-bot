//! Kalshi trading fees. The KXBTC15M series is `fee_type: quadratic`: a taker
//! pays `ceil_to_cent(rate * C * P * (1 - P))` per order.

pub const TAKER_RATE: f64 = 0.07;

pub fn taker_fee(contracts: u32, price: f64, rate: f64) -> f64 {
    let raw = rate * contracts as f64 * price * (1.0 - price);
    // Rounded up to the cent; the epsilon keeps exact cents from bumping.
    ((raw * 100.0) - 1e-9).ceil().max(0.0) / 100.0
}

/// Fee per contract, before rounding: what a strategy should treat as cost.
pub fn fee_per_contract(price: f64, rate: f64) -> f64 {
    rate * price * (1.0 - price)
}

/// The most of `n` contracts that can be sold at `price` without the fee
/// taking cash below zero. The fee rounds up to the cent, so a small sale
/// near 0 can cost more than it brings in.
pub fn max_sell_without_debt(cash: f64, n: u32, price: f64, rate: f64) -> u32 {
    let ok = |k: u32| cash + k as f64 * price - taker_fee(k, price, rate) >= -1e-9;
    if ok(n) {
        return n;
    }
    // Rare (sub-cent prices, almost no cash): take the largest size that works.
    (0..n).rev().find(|k| ok(*k)).unwrap_or(0)
}

/// Kalshi's tapered tick ladder for these markets.
pub fn tick_size(price: f64) -> f64 {
    if !(0.1..0.9).contains(&price) { 0.001 } else { 0.01 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fee_rounds_up_to_the_cent() {
        // 0.07 * 10 * 0.5 * 0.5 = 0.175 -> 0.18
        assert_eq!(taker_fee(10, 0.5, TAKER_RATE), 0.18);
        // 0.07 * 1 * 0.5 * 0.5 = 0.0175 -> 0.02
        assert_eq!(taker_fee(1, 0.5, TAKER_RATE), 0.02);
        // 0.07 * 100 * 0.5 * 0.5 = 1.75 exactly
        assert_eq!(taker_fee(100, 0.5, TAKER_RATE), 1.75);
        assert_eq!(taker_fee(0, 0.5, TAKER_RATE), 0.0);
    }

    #[test]
    fn selling_never_costs_more_than_cash() {
        // 5 contracts at 0.001 bring in half a cent but pay a 1-cent fee.
        assert_eq!(max_sell_without_debt(0.0, 5, 0.001, TAKER_RATE), 0);
        assert_eq!(max_sell_without_debt(0.005, 5, 0.001, TAKER_RATE), 5);
        // 20 at 0.001 bring in 2 cents for a 1-cent fee: fine even from zero.
        assert_eq!(max_sell_without_debt(0.0, 20, 0.001, TAKER_RATE), 20);
        assert_eq!(max_sell_without_debt(0.0, 100, 0.5, TAKER_RATE), 100);
    }

    #[test]
    fn ladder() {
        assert_eq!(tick_size(0.05), 0.001);
        assert_eq!(tick_size(0.5), 0.01);
        assert_eq!(tick_size(0.95), 0.001);
    }
}
