#!/usr/bin/env julia
# coding: utf-8
# author: yuanyeh

import Base.+
import Base.-
import Base.*
import Base.%
import Base.^
import Base.norm
import Base.zero

type Vec
    x::Float64
    y::Float64
    z::Float64
end

const ZERO  = Vec(0., 0., 0.)
const XAXIS = Vec(1., 0., 0.)
const YAXIS = Vec(0., 1., 0.)
const ZAXIS = Vec(0., 0., 1.)

+(a::Vec, b::Vec)     = Vec(a.x + b.x, a.y + b.y, a.z + b.z)
-(a::Vec, b::Vec)     = Vec(a.x - b.x, a.y - b.y, a.z - b.z)
*(a::Vec, b::Vec)     = Vec(a.x * b.x, a.y * b.y, a.z * b.z)
*(a::Vec, b::Float64) = Vec(a.x * b, a.y * b, a.z * b)
*(a::Float64, b::Vec) = Vec(a * b.x, a * b.y, a * b.z)
# dot
%(a::Vec, b::Vec)     = a.x * b.x + a.y * b.y + a.z * b.z
# cross
^(a::Vec, b::Vec)     = Vec(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
zero(a::Vec) = Vec(0., 0., 0.)
norm(a::Vec) = a * (1.0 / sqrt(a.x * a.x + a.y * a.y + a.z * a.z))

type Ray
    o::Vec
    d::Vec
end

const (DIFF, SPEC, REFR) = (1, 2, 3)


type Sphere
    p::Vec           # position
    e::Vec           # emission
    c::Vec           # color
    cc::Vec
    rad::Float64     # radius
    sqRad::Float64   # square of radius
    maxC::Float64
    refl::Int        # reflection type (DIFFuse, SPECular, REFRactive)
end

function newSphere(rad::Float64, p::Vec, e::Vec, c::Vec, refl::Int)
    sqRad = rad * rad
    maxC = c.x > c.y && c.y > c.z ? c.x : c.y > c.z ? c.y : c.z
    return Sphere(p, e, c, c * (1. / maxC), rad, sqRad, maxC, refl)
end

function intersectSphere(s::Sphere, r::Ray)
    # Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    op = s.p - r.o
    b = (op % r.d)
    eps = 1e-4
    det = b * b - (op % op) + s.sqRad
    eps = 1e-4
    if det < 0
        return 1e20
    else
        dets = sqrt(det)

        if b - dets > eps
            return b - dets
        elseif b + dets > eps
            return b + dets
        else
            return 1e20
        end
    end
end

const left = newSphere(1e5,  Vec(1e5+1, 40.8, 81.6),      Vec(0., 0., 0.),    Vec(.75, .25, .25),    DIFF)
const rght = newSphere(1e5,  Vec(-1e5+99., 40.8, 81.6),   Vec(0., 0., 0.),    Vec(.25, .25, .75),    DIFF)
const back = newSphere(1e5,  Vec(50., 40.8, 1e5),         Vec(0., 0., 0.),    Vec(.75, .75, .75),    DIFF)
const frnt = newSphere(1e5,  Vec(50., 40.8, -1e5+170.),   Vec(0., 0., 0.),    Vec(0., 0., 0.),       DIFF)
const botm = newSphere(1e5,  Vec(50., 1e5, 81.6),         Vec(0., 0., 0.),    Vec(.75, .75, .75),    DIFF)
const top  = newSphere(1e5,  Vec(50., -1e5+81.6, 81.6),   Vec(0., 0., 0.),    Vec(.75, .75, .75),    DIFF)
const mirr = newSphere(16.5, Vec(27., 16.5, 47.),         Vec(0., 0., 0.),    Vec(1., 1., 1.)*.999,  SPEC)
const glas = newSphere(16.5, Vec(73., 16.5, 78.),         Vec(0., 0., 0.),    Vec(1., 1., 1.)*.999,  REFR)
const lite = newSphere(600., Vec(50., 681.6-.27, 81.6),   Vec(12., 12., 12.), Vec(0., 0., 0.),       DIFF)

const spheres = [left rght back frnt botm top mirr glas lite]

function clip(x::Float64)
    return max(min(x,one(x)),zero(x))
end

function toInt(x::Float64)
    return floor(Int, clip(x)^(1./2.2) * 255 + .5)
end

function intersectSpheres(r::Ray)
    t = 1e20
    ret = nothing

    for sphere in spheres
        d = intersectSphere(sphere, r)
        if d < t
            t = d
            ret = sphere
        end
    end
    return ret, t
end

function radiance(r::Ray, depth::Int)
    obj, t = intersectSpheres(r)
    if  obj == nothing
        return ZERO
    end

    newDepth = depth + 1
    isMaxDepth = newDepth > 100

    # Russian roulette for path termination
    isUseRR = newDepth > 5
    isRR = isUseRR && rand() < obj.maxC

    if isMaxDepth || (isUseRR && ~isRR)
        return obj.e
    end

    f = (isUseRR && isRR) ? obj.cc : obj.c
    x = r.o + r.d * t
    n = norm(x - obj.p)
    nl = (n % r.d) < 0 ? n : n * -1.

    if obj.refl == DIFF # Ideal DIFFUSE reflection
        r1 = 2 * pi * rand()
        r2 = rand()
        r2s = sqrt(r2)

        w = nl
        wo = w.x < -0.1 || w.x > 0.1 ? YAXIS : XAXIS
        u = norm(wo ^ w)
        v = (w ^ u)

        d = norm(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2))
        return obj.e + f * radiance(Ray(x, d), newDepth)
    elseif obj.refl == SPEC # Ideal SPECULAR reflection
        return obj.e + f * radiance(Ray(x, r.d - n * (2. * (n % r.d))), newDepth)
    else # Ideal dielectric REFRACTION
        reflRay = Ray(x, r.d - n * (2. * (n % r.d)))
        into = (n % nl) > 0
        nc = 1.
        nt = 1.5
        nnt = into ? nc / nt : nt / nc
        ddn = (r.d % nl)
        cos2t = 1 - nnt * nnt * (1 - ddn * ddn)

        if cos2t < 0
            return obj.e + f * radiance(reflRay, newDepth)
        else
            tdir = norm(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))))
            a = nt - nc
            b = nt + nc
            R0 = (a * a) / (b * b)
            c = 1. - (into ? -ddn : (tdir % n))
            Re = R0 + (1 - R0) * c * c * c * c * c
            Tr = 1 - Re
            P = .25 + .5 * Re
            RP = Re / P
            TP = Tr / (1. - P)

            result = Vec(0., 0., 0.)
            if newDepth > 2
                if rand() < P
                    result = radiance(reflRay, newDepth) * RP
                else
                    result = radiance(Ray(x, tdir), newDepth) * TP
                end
            else
                result = radiance(reflRay, newDepth) * Re + radiance(Ray(x, tdir), newDepth) * Tr
            end

            return obj.e + f * result
        end
    end
end

function main()
    # main
    w = 256
    h = 256
    samps = 25
    cam = Ray(Vec(50., 52., 295.6), norm(Vec(0., -0.042612, -1.)))
    cx = Vec(w * .5135 / h, 0., 0.)
    cy = norm(cx ^ cam.d) * .5135
    c = Array{Vec}(w * h)
    for i = 1:length(c)
        c[i] = Vec(0., 0., 0.)
    end

    # Loop over image rows
    for y = 1:h
        # Loop cols
        for x = 1:w
            # 2x2 subpixel rows
            for sy = 1:2
                i = (h - y) * w + x
                for sx = 1:2
                    r = Vec(0., 0., 0.)
                    for s = 1:samps
                        r1 = 2 * rand()
                        r2 = 2 * rand()
                        dx = r1 < 1. ? sqrt(r1) - 1. : 1. - sqrt(2. - r1)
                        dy = r2 < 1. ? sqrt(r2) - 1. : 1. - sqrt(2. - r2)

                        d = cx * (((sx-1 + .5 + dx) / 2 + x) / w - .5) +
                            cy * (((sy-1 + .5 + dy) / 2 + y) / h - .5) + cam.d
                        r = r + radiance(Ray(cam.o + d * 140., norm(d)), 0) * (1. / samps)
                    end
                    c[i] = c[i] + Vec(clip(r.x), clip(r.y), clip(r.z)) * .25
                end
            end
        end
    end

    open("image_julia.ppm", "w") do f
        print(f, "P3\n$w $h\n255\n")
        for i in 1:length(c)
            print(f, toInt(c[i].x))
            print(f, " ")
            print(f, toInt(c[i].y))
            print(f, " ")
            print(f, toInt(c[i].z))
            print(f, "\n")
        end
    end
end

@time main()
