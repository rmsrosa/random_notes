using Documenter

allpages = [
    "Random Notes" => "index.md",
    "Probability" => [
        "Basic things" => [
            "probability/kernel_density_estimation.md"
        ]
    ],
    "Bayesian inference" => [
        "Bayes Theory" => [
            "Bayes Theorem" => "bayesian/bayes.md",
            "Bayesian inference" => "bayesian/bayes_inference.md",
            "Bernstein–von Mises theorem" => "bayesian/bernstein_vonmises.md"
        ],
        "Probabilistic programming" => [
            "Overview" => "bayesian/probprog.md",
            "Markov Chain Monte Carlo (MCMC)" => "bayesian/mcmc.md",
            "Gibbs sampling" => "bayesian/gibbs.md",
            "Hamiltonian Monte Carlo (HMC)" => "bayesian/hmc.md"
        ],
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
            "generative/langevin_sampling.md",
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