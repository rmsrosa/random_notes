# How to docs

## Necessary files

1. Have a `docs/` folder with a `make.jl` file and a `scr/` folder and at least an `index.md` file inside `scr/`;
2. The `make.jl` file should have, at least, `using Documenter` and a `makedocs(sitename="name of site")`.
3. The `index.md` can have whatever.

## Directly with Documenter

1. cd to root directory;
2. activate root project;
3. make sure you have `Documenter.jl` and all the other relevant packages added to the project;
3. create/update the html docs with `> include("docs/make.jl")`;
4. open the indicated local web page in a browser (usually [http://localhost:8000])

## Via LiveServer

1. Have `LiveServer.jl` installed in the global environment or in `/docs`;
2. Do `using LiveServer`;
3. Do `julia> servedocs()`;
4. Wait for the browser to pop-up with the local documentation page.
