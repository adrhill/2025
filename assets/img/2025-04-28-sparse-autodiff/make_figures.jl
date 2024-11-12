using Pkg
Pkg.activate(@__DIR__)
using Luxor
using Colors
using MathTeXEngine
using LaTeXStrings: LaTeXString
using StableRNGs: StableRNG

using Flux: Conv, relu
using DifferentiationInterface: jacobian, AutoForwardDiff
using ForwardDiff: ForwardDiff
using LinearAlgebra: I
using SparseArrays

const purple = Luxor.julia_purple
const red = Luxor.julia_red
const green = Luxor.julia_green
const blue = Luxor.julia_blue

hue_purple = hue(convert(HSL, RGB(Luxor.julia_purple...)))
hue_red = hue(convert(HSL, RGB(Luxor.julia_red...)))
hue_green = hue(convert(HSL, RGB(Luxor.julia_green...)))
hue_blue = hue(convert(HSL, RGB(Luxor.julia_blue...)))

lightness = 0.4
saturation = 0.8

color_purple = HSL(hue_purple, saturation, lightness)
color_red = HSL(hue_red, saturation, 0.45)
color_green = HSL(hue_green, saturation, 0.25)
color_blue = HSL(hue_blue, saturation, lightness)
color_black = convert(HSL, Gray(0))
color_white = convert(HSL, Gray(1))
color_operator = convert(HSL, Gray(0.3))
color_transparent = RGBA(0, 0, 0, 0)
color_background = color_transparent

named_color(name) = convert(HSL, RGB(Colors.color_names[name] ./ 256...))

color_F = color_green
color_H = color_red
color_G = color_purple

CELLSIZE = 20
PADDING = 2  # Increased PADDING for better spacing
FONTSIZE = 18
SPACE = 11

# Function to normalize value between 0 and 1
normalize(x, min, max) = (x - min) / (max - min)
scale(x, min, max, lo, hi) = normalize(x, min, max) * (hi - lo) + lo

abstract type Drawable end
width(D::Drawable) = D.width
height(D::Drawable) = D.height

struct Position{D<:Drawable}
    drawable::D
    center::Point
end
drawable(P::Position) = P.drawable
width(P::Position) = width(drawable(P))
height(P::Position) = height(drawable(P))

center(P::Position) = P.center
xcenter(P::Position) = center(P).x
ycenter(P::Position) = center(P).y

top(P::Position) = Point(xcenter(P), ycenter(P) + height(P) / 2)
bottom(P::Position) = Point(xcenter(P), ycenter(P) - height(P) / 2)
right(P::Position) = Point(xcenter(P) + width(P) / 2, ycenter(P))
left(P::Position) = Point(xcenter(P) - width(P) / 2, ycenter(P))

function position_right_of(P::Position; space = SPACE)
    x, y = right(P)
    function position_drawable(D::Drawable)
        return Position(D, Point(x + space + width(D) / 2, y))
    end
    return position_drawable
end

function position_on(P::Position)
    return function position_drawable(D::Drawable)
        return Position(D, center(P))
    end
end

function draw!(P::Position; offset = Point(0.0, 0.0))
    center = P.center + offset
    draw!(P.drawable, center)
    return nothing
end

#========#
# Matrix #
#========#

default_cell_text(x) = string(round(x; digits = 2))
Base.@kwdef struct DrawMatrix <: Drawable
    mat::Matrix{Float64}
    color::HSL{Float64} = color_black
    cellsize::Float64 = CELLSIZE
    padding_inner::Float64 = PADDING
    padding_outer::Float64 = 1.75 * PADDING
    border_inner::Float64 = 0.75
    border_outer::Float64 = 2.0
    dashed::Bool = false
    show_text::Bool = false
    mat_text::Matrix{String} = map(default_cell_text, mat)
    absmax::Float64 = maximum(abs, mat)
    height::Float64 =
        size(mat, 1) * (cellsize + padding_inner) - padding_inner + 2 * padding_outer
    width::Float64 =
        size(mat, 2) * (cellsize + padding_inner) - padding_inner + 2 * padding_outer
    column_colors = fill(color, size(mat, 2))
end

function draw!(M::DrawMatrix, center::Point)
    # Destructure DrawMatrix for convenience
    (;
        mat,
        color,
        cellsize,
        padding_inner,
        padding_outer,
        border_inner,
        border_outer,
        dashed,
        show_text,
        mat_text,
        absmax,
        height,
        width,
        column_colors,
    ) = M

    rows, cols = size(mat)

    # Apply offset
    xcenter, ycenter = center
    # Compute upper left edge of matrix
    x0 =
        xcenter - (cols / 2) * (cellsize + padding_inner) + padding_inner / 2 -
        padding_outer
    y0 =
        ycenter - (rows / 2) * (cellsize + padding_inner) + padding_inner / 2 -
        padding_outer

    setline(1)
    for i = 1:rows
        for j = 1:cols
            # Calculate cell position (corner of matrix entry)
            x = x0 + (j - 1) * (cellsize + padding_inner) + padding_outer
            y = y0 + (i - 1) * (cellsize + padding_inner) + padding_outer

            # Calculate color based on normalized value
            val = mat[i, j]
            h = hue(column_colors[j])
            v = 0.8
            l = scale(abs(val), 0, absmax, 1.0, 0.25)
            cell_color = HSL(h, v, l)

            # Draw rectangle
            setcolor(cell_color)
            rect(Point(x, y), cellsize, cellsize, :fill)

            # Draw border
            setline(border_inner)
            setcolor(column_colors[j])
            iszero(val) && setcolor("lightgray")
            rect(Point(x, y), cellsize, cellsize, :stroke)

            # Add text showing matrix value
            if show_text
                fontsize(min(cellsize ÷ 3, 14))
                if !is_background_bright(cell_color)
                    setcolor(color_white)
                end
                text(
                    mat_text[i, j],
                    Point(x + cellsize / 2, y + cellsize / 2);
                    halign = :center,
                    valign = :middle,
                )
            end
        end
    end

    # Draw border
    setline(border_outer)
    setcolor(color)
    dashed && setdash([7.0, 4.0])
    setlinejoin("miter")
    rect(Point(x0, y0), width, height, :stroke)
    return setdash("solid")
end

is_background_bright(bg) = is_background_bright(convert(RGB, bg))
is_background_bright(bg::RGB) = luma(bg) > 0.5
luma(c::RGB) = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b # using BT. 709 coefficients

#==========#
# Operator #
#==========#

Base.@kwdef struct DrawOperator <: Drawable
    text::LaTeXString
    color::HSL{Float64} = color_operator
    cellsize::Float64 = 12
    fontsize::Float64 = 20
end
width(O::DrawOperator) = O.cellsize
height(O::DrawOperator) = O.cellsize

function draw!(O::DrawOperator, center)
    # Apply offset
    setcolor(O.color)
    fontsize(O.fontsize)
    return text(
        O.text,
        center - Point(0.15 * O.cellsize, 0.0);
        halign = :center,
        valign = :middle,
    )
end

#==========#
# Operator #
#==========#

Base.@kwdef struct DrawOverlay <: Drawable
    text::LaTeXString
    color::HSL{Float64} = color_operator
    background::HSL{Float64} = HSL(color.h, color.s, 0.9)
    width::Float64 = 50
    height::Float64 = 33
    radius::Float64 = 12
    fontsize::Float64 = 20
end

function draw!(O::DrawOverlay, center)
    # Apply offset
    setcolor(O.background)
    box(center, O.width, O.height, O.radius; action = :fill)
    setline(0.75)
    setcolor(O.color)
    box(center, O.width, O.height, O.radius; action = :stroke)
    fontsize(O.fontsize)
    return text(O.text, center + Point(0, 2); halign = :center, valign = :middle)
end

#==========#
# Draw PDF #
#==========#

# Get random matrices
n, m, p = 4, 5, 3
H = randn(StableRNG(121), n, p)
G = randn(StableRNG(123), p, m)
F = H * G

S = Matrix(
    [
        0.0 -2.295 0.0 0.207 0.0
        0.0 0.0 0.0 0.170 2.11
        0.0 1.852 1.472 0.0 0.0
        -0.479 0.0 -0.264 0.0 0.0
    ],
)
iszero_string(x) = !iszero(x) ? "≠ 0" : "0"

P = map(!iszero, S)
P_text = map(iszero_string, P)

vFr = randn(StableRNG(3), m, 1)
vHr = G * vFr
vRr = H * vHr # result from right

# Create drawables
DF = DrawMatrix(; mat = F, color = color_F)
DG = DrawMatrix(; mat = G, color = color_G)
DH = DrawMatrix(; mat = H, color = color_H)

DFd = DrawMatrix(; mat = F, color = color_F, dashed = true)
DGd = DrawMatrix(; mat = G, color = color_G, dashed = true)
DHd = DrawMatrix(; mat = H, color = color_H, dashed = true)

DFdn = DrawMatrix(; mat = F, color = color_F, dashed = true, show_text = true)

DEq = DrawOperator(; text = "=")
DTimes = DrawOperator(; text = "⋅")
DCirc = DrawOperator(; text = "∘")

DJF = DrawOverlay(; text = L"J_{f}(x)", color = color_F)
DJG = DrawOverlay(; text = L"J_{g}(x)", color = color_G)
DJH = DrawOverlay(; text = L"J_{h}(g(x))", color = color_H, fontsize = 18, width = 65)

DDF = DrawOverlay(; text = "Df(x)", color = color_F, fontsize = 18)
DDG = DrawOverlay(; text = "Dg(x)", color = color_G, fontsize = 18)
DDH = DrawOverlay(; text = "Dh(g(x))", color = color_H, fontsize = 15, width = 65)

function setup!()
    background(color_background)
    fontsize(18)
    fontface("JuliaMono")
end

function chainrule(; show_text = true)
    setup!()

    DFn = DrawMatrix(; mat = F, color = color_F, show_text = show_text)
    DGn = DrawMatrix(; mat = G, color = color_G, show_text = show_text)
    DHn = DrawMatrix(; mat = H, color = color_H, show_text = show_text)

    # Position drawables
    drawables = [DFn, DEq, DHn, DTimes, DGn]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DF) - total_width) / 2
    ystart = 0.0

    PF = Position(DFn, Point(xstart, ystart))
    PEq = position_right_of(PF)(DEq)
    PH = position_right_of(PEq)(DHn)
    PTimes = position_right_of(PH)(DTimes)
    PG = position_right_of(PTimes)(DGn)

    PJF = position_on(PF)(DJF)
    PJG = position_on(PG)(DJG)
    PJH = position_on(PH)(DJH)

    # Draw 
    for obj in (PF, PG, PH, PEq, PTimes, PJF, PJG, PJH)
        draw!(obj)
    end
end

# Draw the Jacobian of the first layer of the small LeNet-5 CNN
function big_conv_jacobian()
    setup!()
    layer = Conv((5, 5), 1 => 1, identity)
    input = randn(Float32, 28, 28, 1, 1)

    J = jacobian(layer, AutoForwardDiff(), input)
    # @info "Size of the Conv Jacobian:" size(J) relative_sparsity=sum(iszero,J)/length(J)

    DJ = DrawMatrix(;
        mat = J,
        color = color_G,
        cellsize = 2,
        padding_inner = 0,
        padding_outer = 0,
        border_inner = 0,
        border_outer = 10,
    )
    DJF = DrawOverlay(;
        text = L"J_{g}(x)",
        color = color_G,
        fontsize = 150,
        width = 350,
        height = 180,
    )

    center = Point(0.0, 0.0)
    PJ = Position(DJ, center)
    PDJ = Position(DJF, center)
    draw!(PJ)
    return draw!(PDJ)
end

function matrixfree()
    setup!()

    # Position drawables
    drawables = [DFd, DEq, DHd, DCirc, DGd]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DF) - total_width) / 2
    ystart = 0.0

    PF = Position(DFd, Point(xstart, ystart))
    PEq = position_right_of(PF)(DEq)
    PH = position_right_of(PEq)(DHd)
    PCirc = position_right_of(PH)(DCirc)
    PG = position_right_of(PCirc)(DGd)

    PDF = position_on(PF)(DDF)
    PDG = position_on(PG)(DDG)
    PDH = position_on(PH)(DDH)

    # Draw 
    for obj in (PF, PG, PH, PEq, PCirc, PDF, PDG, PDH)
        draw!(obj)
    end
end

function matrixfree2()
    setup!()

    DvFr = DrawMatrix(; mat = vFr, color = color_blue)
    DvHr = DrawMatrix(; mat = vHr, color = color_blue)
    DvRr = DrawMatrix(; mat = vRr, color = color_blue)

    # Position drawables
    drawables = [DFd, DvFr, DEq, DHd, DGd, DvFr]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DF) - total_width) / 2
    ystart = -105.0

    PF = Position(DFd, Point(xstart, ystart))
    PvFr = position_right_of(PF)(DvFr)

    PEq = position_right_of(PvFr)(DEq)
    PH = position_right_of(PEq)(DHd)
    PG = position_right_of(PH)(DGd)
    PvFr2 = position_right_of(PG)(DvFr)

    PEq2 = Position(DEq, center(PEq) + Point(0, 110))
    PH2 = position_right_of(PEq2)(DHd)
    PvHr = position_right_of(PH2)(DvHr)

    PEq3 = Position(DEq, center(PEq2) + Point(0, 110))
    PvRr = position_right_of(PEq3)(DvRr)

    PDF = position_on(PF)(DDF)
    PDG = position_on(PG)(DDG)
    PDH = position_on(PH)(DDH)
    PDH2 = position_on(PH2)(DDH)

    # Draw 
    for obj in
        (PF, PvFr, PEq, PH, PG, PvFr2, PEq2, PH2, PvHr, PEq3, PvRr, PDF, PDG, PDH, PDH2)
        draw!(obj)
    end
end

function forward_mode()
    setup!()

    i1 = 1
    e1 = zeros(m, 1)
    e1[i1, 1] = 1

    i2 = 5
    e2 = zeros(m, 1)
    e2[i2, 1] = 1

    absmax = maximum(abs, F)

    F1_text = map(default_cell_text, F)
    F2_text = map(default_cell_text, F)
    F1_text[:, 2:end] .= ""
    F2_text[:, begin:(end-1)] .= ""

    DF1 = DrawMatrix(;
        mat = F,
        mat_text = F1_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )
    DF2 = DrawMatrix(;
        mat = F,
        mat_text = F2_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )
    De1 = DrawMatrix(; mat = e1, color = color_blue, show_text = true)
    De2 = DrawMatrix(; mat = e2, color = color_blue, show_text = true)
    DFe1 = DrawMatrix(; mat = F * e1, color = color_F, absmax = absmax, show_text = true)
    DFe2 = DrawMatrix(; mat = F * e2, color = color_F, absmax = absmax, show_text = true)
    DDots = DrawOperator(; text = "...", fontsize = 40)

    # Position drawables
    drawables = [DF1, De1, DEq, DFe1, DDots, DFdn, De2, DEq, DFe2]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE + 40
    xstart = (width(DF1) - total_width) / 2
    ystart = 0.0

    PF1 = Position(DF1, Point(xstart, ystart))
    Pe1 = position_right_of(PF1)(De1)
    PEq1 = position_right_of(Pe1)(DEq)
    PFe1 = position_right_of(PEq1)(DFe1)

    PDots = position_right_of(PFe1; space = 30)(DDots)

    PF2 = position_right_of(PDots; space = 30)(DF2)
    Pe2 = position_right_of(PF2)(De2)
    PEq2 = position_right_of(Pe2)(DEq)
    PFe2 = position_right_of(PEq2)(DFe2)

    PDF1 = position_on(PF1)(DDF)
    PDF2 = position_on(PF2)(DDF)

    # Draw 
    for obj in (PF1, Pe1, PEq1, PFe1, PF2, Pe2, PEq2, PFe2, PDF1, PDF2, PDots)
        draw!(obj)
    end
end

function reverse_mode()
    setup!()

    i1 = 1
    e1 = zeros(1, n)
    e1[1, i1] = 1

    i2 = 4
    e2 = zeros(1, n)
    e2[1, i2] = 1

    absmax = maximum(abs, F)

    F1_text = map(default_cell_text, F)
    F2_text = map(default_cell_text, F)
    F1_text[2:end, :] .= ""
    F2_text[begin:(end-1), :] .= ""

    DF1 = DrawMatrix(;
        mat = F,
        mat_text = F1_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )
    DF2 = DrawMatrix(;
        mat = F,
        mat_text = F2_text,
        color = color_F,
        dashed = true,
        show_text = true,
    )

    De1 = DrawMatrix(; mat = e1, color = color_blue, show_text = true)
    De2 = DrawMatrix(; mat = e2, color = color_blue, show_text = true)
    DFe1 = DrawMatrix(; mat = e1 * F, color = color_F, absmax = absmax, show_text = true)
    DFe2 = DrawMatrix(; mat = e2 * F, color = color_F, absmax = absmax, show_text = true)
    DDots = DrawOperator(; text = "...", fontsize = 40)

    # Position drawables
    drawables = [De1, DF1, DEq, DFe1]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(De1) - total_width) / 2
    ystart = -65.0

    Pe1 = Position(De1, Point(xstart, ystart))
    PF1 = position_right_of(Pe1)(DF1)
    PEq1 = position_right_of(PF1)(DEq)
    PFe1 = position_right_of(PEq1)(DFe1)

    PDots = Position(DDots, Point(0, ystart + 71.0))

    Pe2 = Position(De2, Point(xstart, ystart + 140.0))
    PF2 = position_right_of(Pe2)(DF2)
    PEq2 = position_right_of(PF2)(DEq)
    PFe2 = position_right_of(PEq2)(DFe2)

    PDF1 = position_on(PF1)(DDF)
    PDF2 = position_on(PF2)(DDF)

    # Draw 
    for obj in (PF1, Pe1, PEq1, PFe1, PF2, Pe2, PEq2, PFe2, PDF1, PDF2, PDots)
        draw!(obj)
    end
end

function sparsity(; ismap = false)
    setup!()
    DS = DrawMatrix(; mat = S, color = color_F, dashed = ismap, show_text = !ismap)
    PS = Position(DS, Point(0.0, 0.0))
    return draw!(PS)
end

function sparse_map_colored()
    setup!()

    c1 = named_color("orchid")
    c2 = named_color("lightslateblue")
    column_colors = [c1, c1, c2, c2, c1]

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        column_colors = column_colors,
    )
    PS = Position(DS, Point(0.0, 0.0))
    return draw!(PS)
end

function sparsity_pattern()
    setup!()

    P_text = map(x -> !iszero(x) ? "≠ 0" : "0", P)
    DP = DrawMatrix(; mat = P, mat_text = P_text, color = color_F, show_text = true)
    PP = Position(DP, Point(0.0, 0.0))
    return draw!(PP)
end

function sparsity_coloring()
    setup!()

    c1 = named_color("orchid")
    c2 = named_color("lightslateblue")
    column_colors = [c1, c1, c2, c2, c1]

    DP = DrawMatrix(;
        mat = P,
        mat_text = P_text,
        color = color_F,
        show_text = true,
        column_colors = column_colors,
    )
    PP = Position(DP, Point(0.0, 0.0))
    return draw!(PP)
end

function sparse_ad()
    setup!()

    v = reshape([1.0 1.0 0.0 0.0 1.0], 5, 1)
    absmax = maximum(abs, S)

    c1 = named_color("orchid")
    c2 = named_color("lightslateblue")
    column_colors = [c1, c1, c2, c2, c1]

    DS = DrawMatrix(;
        mat = S,
        color = color_F,
        dashed = true,
        show_text = true,
        column_colors = column_colors,
    )
    Dv = DrawMatrix(; mat = v, color = color_blue, show_text = true)
    DSv = DrawMatrix(;
        mat = S * v,
        color = color_F,
        absmax = absmax,
        show_text = true,
        column_colors = [c1],
    )

    # Position drawables
    drawables = [DS, Dv, DEq, DSv]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DS) - total_width) / 2
    ystart = 0.0

    PS = Position(DS, Point(xstart, ystart))
    Pv = position_right_of(PS)(Dv)
    PEq = position_right_of(Pv)(DEq)
    PSv = position_right_of(PEq)(DSv)

    # Draw 
    for obj in (PS, Pv, PEq, PSv)
        draw!(obj)
    end
end

function sparsity_pattern_compressed()
    setup!()

    P = fill(1.0, n, 1)
    P_text = reshape(["{2,4}", "{4,5}", "{2,3}", "{1,3}"], n, 1)
    DP = DrawMatrix(; mat = P, mat_text = P_text, color = color_F, show_text = true)
    PP = Position(DP, Point(0.0, 0.0))
    return draw!(PP)
end

function forward_mode_naive()
    setup!()

    DFd = DrawMatrix(; mat = S, color = color_F, dashed = true, show_text = false)
    DI = DrawMatrix(; mat = I(5), color = color_blue, show_text = true)
    DFj = DrawMatrix(; mat = S * I(5), color = color_F, show_text = true)

    # Position drawables
    drawables = [DFd, DI, DEq, DFj]
    total_width = sum(width, drawables) + (length(drawables) - 1) * SPACE
    xstart = (width(DFd) - total_width) / 2
    ystart = 0.0

    PFd = Position(DFd, Point(xstart, ystart))
    PI = position_right_of(PFd)(DI)
    PEq = position_right_of(PI)(DEq)
    PFj = position_right_of(PEq)(DFj)
    @info PFd.center # Point(-137.5, 0.0)

    PDF = position_on(PFd)(DDF)
    PJF = position_on(PFj)(DJF)

    # Draw 
    for obj in (PFd, PI, PEq, PFj, PDF, PJF)
        draw!(obj)
    end
end

function forward_mode_sparse()
    setup!()

    PI = fill(1, m, 1)
    PJ = fill(1, n, 1)

    PI_text = reshape(["{1}", "{2}", "{3}", "{4}", "{5}"], m, 1)
    PJ_text = reshape(["{2,4}", "{4,5}", "{2,3}", "{1,3}"], n, 1)
    P_text = map(x -> !iszero(x) ? "≠ 0" : "0", P)

    DFd = DrawMatrix(; mat = S, color = color_F, dashed = true, show_text = false)
    DI = DrawMatrix(; mat = PI, mat_text = PI_text, color = color_blue, show_text = true)
    DFj = DrawMatrix(; mat = PJ, mat_text = PJ_text, color = color_F, show_text = true)
    DP = DrawMatrix(; mat = P, mat_text = P_text, color = color_F, show_text = true)

    DEq2 = DrawOperator(; text = "≔")

    # Position drawables
    PFd = Position(DFd, Point(-137.5, 0.0)) # reuse center from `forward_mode_naive`
    PI = position_right_of(PFd)(DI)
    PEq1 = position_right_of(PI)(DEq)
    PFj = position_right_of(PEq1)(DFj)
    PEq2 = position_right_of(PFj)(DEq2)
    PP = position_right_of(PEq2)(DP)

    PDF = position_on(PFd)(DDF)

    # Draw 
    for obj in (PFd, PI, PEq1, PFj, PEq2, PP, PDF)
        draw!(obj)
    end
end

# This one is huge, avoid SVG and PDF:
@png big_conv_jacobian() 1600 1200 joinpath(@__DIR__, "big_conv_jacobian")

# Change the default saving format here
var"@save" = var"@svg" # var"@pdf"

@save chainrule() 380 100 joinpath(@__DIR__, "chainrule")
@save chainrule(; show_text = true) 380 100 joinpath(@__DIR__, "chainrule_num")
@save matrixfree() 380 100 joinpath(@__DIR__, "matrixfree")
@save matrixfree2() 450 340 joinpath(@__DIR__, "matrixfree2")

@save forward_mode() 510 120 joinpath(@__DIR__, "forward_mode")
@save reverse_mode() 380 250 joinpath(@__DIR__, "reverse_mode")

@save sparsity() 120 100 joinpath(@__DIR__, "sparse_matrix")
@save sparsity(; ismap = true) 120 100 joinpath(@__DIR__, "sparse_map")

@save sparse_ad() 220 120 joinpath(@__DIR__, "sparse_ad")
@save sparse_map_colored() 120 100 joinpath(@__DIR__, "sparse_map_colored")

@save sparsity_pattern() 120 100 joinpath(@__DIR__, "sparsity_pattern")
@save sparsity_coloring() 120 100 joinpath(@__DIR__, "coloring")
@save sparsity_pattern_compressed() 40 100 joinpath(@__DIR__, "sparsity_pattern_compressed")

# Sized need to match:
@save forward_mode_naive() 400 120 joinpath(@__DIR__, "forward_mode_naive")
@save forward_mode_sparse() 400 120 joinpath(@__DIR__, "forward_mode_sparse")