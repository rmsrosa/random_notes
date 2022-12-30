using Documenter

makedocs(
    sitename = "Random notes",
    pages = [
        "Random Notes" => "index.md",
        "Probability" => [
            "Bayes Theorem" => "probability/bayes.md",
            "Bayesian inference" => "probability/bayes_inference.md",
            "Bernstein–von Mises theorem" => "probability/bernstein_vonmises.md",
            "Probabilistic programming" => "probability/probprog.md",
            "Markov Chain Monte Carlo (MCMC)" => "probability/mcmc.md",
            "Many Ways to Linear Regression" => "probability/linear_regression.md",
            "Estimating π via frequentist and Bayesian methods" => "probability/find_pi.md",
            "Modeling mortality tables" => "probability/mortality_tables.md"
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