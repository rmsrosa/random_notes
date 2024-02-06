# How to docs

## Necessary files

1. Have a `docs/` folder with a `make.jl` file and a `scr/` folder and at least an `index.md` file inside `scr/`;
2. The `make.jl` file should have, at least, `using Documenter` and a `makedocs(sitename="name of site")`.
3. The `index.md` can have whatever.

## Directly with Documenter

1. cd to root directory;
2. activate docs project with `pkg> activate docs`;
3. make sure you have `Documenter.jl` and all the other relevant packages added to the docs project;
3. create/update the html docs with `julia> include("docs/make.jl")`;
4. open the indicated local web page in a browser (usually [http://localhost:8000])

## Via LiveServer

1. Have `LiveServer.jl` installed in the global environment or in `/docs`;
2. Activate `docs` project with e.g. `pkg> activate docs` from the root directory
3. Do `julia> using LiveServer`;
4. Do `julia> servedocs()`;
5. Wait for the browser to pop-up with the local documentation page.

## Development

`Documenter.jl` renders every page each time it runs, including when using it via LiveServer. That gets a bit heavy and slow. Hence, for development, it is interesting to render only the page you are currently working on, as follows.

1. Set `draft = true,` in `docs/make.jl`
2. Add a `@meta` block with `Draft = false` somewhat in the beginning of the page you are working on.
3. Don't forget to revert to `draft = true` in `docs/make.jl` before your final commit.
