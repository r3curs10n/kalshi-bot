//! Request signing: RSA-PSS/SHA256 over `timestamp_ms + METHOD + path`, where
//! path excludes the query string. The websocket handshake is signed as a GET
//! of its path.

use anyhow::{Context, Result};
use base64::Engine;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::BlindedSigningKey;
use rsa::rand_core::OsRng;
use rsa::signature::{RandomizedSigner, SignatureEncoding};
use rsa::RsaPrivateKey;
use sha2::Sha256;

pub struct Signer {
    key_id: String,
    // Salt length defaults to the digest size (32), matching Kalshi's examples.
    key: BlindedSigningKey<Sha256>,
}

impl Signer {
    pub fn from_pem(key_id: impl Into<String>, pem: &str) -> Result<Self> {
        let key = RsaPrivateKey::from_pkcs1_pem(pem)
            .or_else(|_| RsaPrivateKey::from_pkcs8_pem(pem))
            .context("private key is neither PKCS#1 nor PKCS#8 PEM")?;
        Ok(Self { key_id: key_id.into(), key: BlindedSigningKey::new(key) })
    }

    pub fn sign(&self, timestamp_ms: i64, method: &str, path: &str) -> String {
        let message = format!("{timestamp_ms}{method}{path}");
        let signature = self.key.sign_with_rng(&mut OsRng, message.as_bytes());
        base64::engine::general_purpose::STANDARD.encode(signature.to_bytes())
    }

    /// The three auth headers for a request to `path` (no query string).
    pub fn headers(&self, method: &str, path: &str) -> [(&'static str, String); 3] {
        let ts = crate::now_ms();
        [
            ("KALSHI-ACCESS-KEY", self.key_id.clone()),
            ("KALSHI-ACCESS-SIGNATURE", self.sign(ts, method, path)),
            ("KALSHI-ACCESS-TIMESTAMP", ts.to_string()),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsa::pkcs1::EncodeRsaPrivateKey;
    use rsa::pss::{Signature, VerifyingKey};
    use rsa::signature::Verifier;

    #[test]
    fn signature_verifies_over_timestamp_method_path() {
        let key = RsaPrivateKey::new(&mut OsRng, 2048).unwrap();
        let pem = key.to_pkcs1_pem(Default::default()).unwrap();
        let signer = Signer::from_pem("id", &pem).unwrap();

        let sig_b64 = signer.sign(1700000000000, "GET", "/trade-api/v2/portfolio/balance");
        let raw = base64::engine::general_purpose::STANDARD.decode(sig_b64).unwrap();
        let verifier = VerifyingKey::<Sha256>::new(key.to_public_key());
        let sig = Signature::try_from(raw.as_slice()).unwrap();
        verifier
            .verify(b"1700000000000GET/trade-api/v2/portfolio/balance", &sig)
            .expect("signature must verify");
        assert!(verifier.verify(b"1700000000001GET/trade-api/v2/portfolio/balance", &sig).is_err());
    }
}
