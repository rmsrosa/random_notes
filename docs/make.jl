using Documenter

makedocs(
    sitename = "Random notes",
    pages = [
        "Welcome" => "index.md",
        "Probability" => [
            "Bayes" => "probability/bayes.md",
            "MCMC" => "probability/mcmc.md",
            "Linear Regression" => "probability/regression.md"
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