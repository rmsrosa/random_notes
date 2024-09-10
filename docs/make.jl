using Documenter

allpages = [
    "Random Notes" => "index.md",
    "Probability Essentials" => [
        "Kernel Density Estimation" => "probability/kernel_density_estimation.md",
        "probability/convergence_notions.md",
    ],
    "Discrete-time Markov chains" => [
        "Essential definitions" => "markov_chains/mc_definitions.md",
        "markov_chains/mc_invariance.md",
        "Countable-space Markov chains" => [
            "markov_chains/mc_countableX_recurrence.md",
            "markov_chains/mc_countableX_connections.md",
            "markov_chains/mc_countableX_convergencia.md"
        ],
        "Continuous-space Markov chains" => [
            "markov_chains/mc_irreducibility_and_recurrence.md",
        ],
    ],
    "Sampling methods" => [
        "Overview" => "sampling/overview.md",
        "Random number generators" => "sampling/prng.md",
        "Transform methods" => [
            "Probability integral transform" => "sampling/invFtransform.md",
            "Box-Muller transform" => "sampling/box_muller.md",
        ],
        "Accept-Reject methods" => [
            "Rejection sampling" => "sampling/rejection_sampling.md",
            "Empirical supremum rejection sampling" => "sampling/empiricalsup_rejection.md"
        ],
        "Markov Chain Monte Carlo (MCMC)" => [
            "Overview" => "sampling/mcmc.md",
            "Metropolis and Metropolis-Hastings" => "sampling/metropolis.md",
            "Convergence of Metropolis-Hastings" => "sampling/convergence_metropolis.md",
            "Gibbs sampling" => "sampling/gibbs.md",
            "Hamiltonian Monte Carlo (HMC)" => "sampling/hmc.md"
        ],
        "sampling/langevin_sampling.md",
    ],
    "Bayesian inference" => [
        "Bayes Theory" => [
            "Bayes Theorem" => "bayesian/bayes.md",
            "Bayesian inference" => "bayesian/bayes_inference.md",
            "Bernstein–von Mises theorem" => "bayesian/bernstein_vonmises.md"
        ],
        "Bayesian probabilistic programming" => "bayesian/bayesian_probprog.md",
        "Examples" => [
            "Estimating π via frequentist and Bayesian methods" => "bayesian/find_pi.md",
            "Many Ways to Linear Regression" => "bayesian/linear_regression.md",
            "Alometry law for the Nile Tilapia" => "bayesian/tilapia_alometry.md",
            "Modeling mortality tables" => "bayesian/mortality_tables.md"
        ]
    ],
    "Generative models" => [
        "Score matching" => [
            "Overview" => "generative/overview.md",
            "generative/stein_score.md",
            "generative/score_matching_aapo.md",
            "generative/score_matching_neural_network.md",
            "generative/parzen_estimation_score_matching.md",
            "generative/denoising_score_matching.md",
            "generative/sliced_score_matching.md",
            "1D finite-difference score matching" => "generative/1d_FD_score_matching.md",
            "2D finite-difference score matching" => "generative/2d_FD_score_matching.md",
            "generative/ddpm.md",
            "Multiple denoising score matching" => "generative/mdsm.md",
            "Probability flow" => "generative/probability_flow.md",
            "Reverse probability flow" => "generative/reverse_flow.md",
            "Score-based SDE model" => "generative/score_based_sde.md",
        ]
    ],
    "Sensitivity analysis" => [
        "Overview" => "sensitivity/overview.md"
    ]
]

makedocs(
    sitename = "Random notes",
    pages = allpages,
    pagesonly = true,
    draft = "draft" in ARGS,
    authors = "Ricardo Rosa",
    format = Documenter.HTML(;
        assets = [
                asset("assets/style.css", islocal=true),
                asset("assets/favicon.ico", islocal=true),
            ],
        prettyurls = get(ENV, "CI", "false") == "true",
        sidebar_sitename = true,
        canonical = "https://github.com/rmsrosa/random_notes",
        edit_link = "main",
        inventory_version = "0.1.0"
    ),
)

if get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
    deploydocs(
        repo = "github.com/rmsrosa/random_notes.jl.git",
        devbranch = "main",
        forcepush = true,
    )
end