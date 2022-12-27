using Documenter

makedocs(
    sitename = "Random notes",
    pages = [
        "Random Notes" => "index.md",
        "Probability" => [
            "Bayes Theorem" => "probability/bayes.md",
            "Bayesian inference" => "probability/bayes_inference.md",
            "Probabilistic programming" => "probability/probprog.md",
            "Markov Chain Monte Carlo (MCMC)" => "probability/mcmc.md",
            "Many Ways to Linear Regression" => "probability/regression.md"
        ],
        "Sensitivity analysis" => [
            "Overview" => "sensitivity/overview.md"
        ]
    ],
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
    ),
)