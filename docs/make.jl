using Documenter, GameTracing

makedocs(;
    sitename = "GameTracing.jl",
    authors = "Carlos Parada",
    modules = [GameTracing],
    format = Documenter.HTML(canonical = "https://paradacarleton.github.io/GameTracing/stable"),
    pages = ["Home" => "index.md", "API" => "api.md"],
    warnonly = false,
)

deploydocs(repo = "github.com/ParadaCarleton/GameTracing.git", push_preview = true)
