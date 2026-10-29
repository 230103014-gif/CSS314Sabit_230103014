{
  "format": "local-verifier-sha256-v1",
  "student_id": "230103014",
  "verified_at_utc": "2026-10-29T07:52:09.302873+00:00",
  "smoke": false,
  "numerical_checks_passed": true,
  "observations": {
    "stride8_to_stride1_latency_ratio": 1.3341022907118436,
    "unpadded_T8_over_T1": 2.2719907758777884,
    "unpadded_over_padded_T8": 1.3771394925263167
  },
  "warnings": [
    "Theoretical CPU peaks not supplied: Roofline regime remains conditional."
  ],
  "implemented_evidence_checks_passed": false,
  "git_history_checked": false,
  "submission_requirements_all_confirmed": false,
  "verification_mode": "regenerate actual kernels and independent references; compare originally recorded salted digests; original timings preserved",
  "file_sha256": {
    "false_sharing_karp_flatt.tsv": "81f26ce78fdb9fa6f806a142d570d4fb540079047acdb1943e3b992f3bdcd43a",
    "host_profile.json": "76102009f4b213d618bbc458d339c280dbe2837fb7cdfae6ad2c52f237475eac",
    "loop_interchange.json": "860515c3596e778f52822d570c99cc1028268c1cadc8bcaa50f78d9b01829dc1",
    "roofline_summary.json": "db6df535b72f5dd8b0eb5cd9483c774a890d8edadd091e100ee61f05ea5f82a4",
    "salted_params.json": "b10368f3a7d298af97cf2e3013d91867ac453a887aa100ed2a2cac7391b7b250",
    "scaling_metrics.tsv": "5c484aed0c9393f81d8b3e2a9611dcaa47c0129c6baa94d1012430a4a2b60e61",
    "stride_results.tsv": "25e44504a8f36b5a3444a255bd5d03e73290e941e8e6ea11abed58782423e810"
  },
  "milestone_tags_checked": [],
  "source_sha256": {
    "src/dense_kernel.py": "d4f527f4ed954325a138081db6befcb8f616f8151281bd63add7f8ac16100f6e",
    "src/false_sharing.py": "f4befb2fb10750bb659a068b5423e6f88a713d7b7ffcda34ab4ed144535a5864",
    "src/profile_env.py": "87d496107a58460268e703d0edadd76d20e94530ca9d8d7902b05ed97c87f9e9",
    "src/salt.py": "dd4b96c67209cdbd74a5fa1bd2fc671b715a2597673a1602352811d7c352e991",
    "src/scaling_study.py": "7e49e58ed50bdef74167c27341383c4c1137c1797358c2189699db8671ab955d",
    "src/stride_bench.py": "5ae6df7a84ab298a2fdc4343b55d96542209e72cc70e81b85dd259cc76711d23"
  },
  "authenticity_note": "Locally generated integrity digest, not a digital signature or instructor-issued verification. Does not authenticate authorship or timing.",
  "receipt_sha256": "4cb26483d712310bde0ebb8236c6829a91e9542ce2b4674e90514592241b9dd5"
}
