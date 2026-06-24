# Rust

Rust is a statically typed, compiled systems programming language designed for performance, reliability, and memory safety. It aims to give programmers the low-level control of C and C++ while eliminating whole categories of bugs at compile time, especially the memory and concurrency errors that have caused decades of security vulnerabilities in systems software.

## History

Rust began as a personal project of Graydon Hoare, a Mozilla employee, around 2006. Mozilla began sponsoring the project in 2009 and announced it publicly in 2010. After years of rapid and often breaking changes, the language reached its first stable release, Rust 1.0, in 2015. Since then Rust has followed a six-week release train, shipping a new stable version regularly while preserving backward compatibility through an "editions" mechanism that lets the language evolve without breaking existing code. In 2021 the Rust Foundation was created, with Mozilla, Google, Microsoft, Amazon, and others as founding members, moving stewardship of the language away from any single company.

## Ownership and memory safety

Rust's most distinctive feature is its ownership system. Every value has a single owner, and when the owner goes out of scope the value is automatically freed. Values can be borrowed, either immutably by many readers at once or mutably by exactly one writer, but never both at the same time. A component of the compiler called the borrow checker enforces these rules statically, before the program ever runs. The result is memory safety without a garbage collector: no use-after-free, no double frees, and no data races, all guaranteed at compile time with no runtime overhead. This is what people mean when they call Rust's abstractions "zero-cost" — the safety checks happen during compilation and disappear from the running binary.

## Type system and tooling

Rust is statically typed with type inference, so the compiler usually figures out types without explicit annotations. It compiles to native machine code through an LLVM backend, giving performance comparable to C and C++. The language has no null and no exceptions; instead it uses the Option type to represent presence or absence and the Result type to represent success or failure, forcing callers to handle error cases explicitly. Pattern matching, traits for shared behavior, and generics round out the type system. Rust's tooling is widely admired: Cargo is the build system and package manager, crates.io hosts the community's libraries, and rustfmt and Clippy handle formatting and linting out of the box.

## Use cases

Rust is commonly used for systems programming, command-line tools, network services, embedded devices, game engines, and WebAssembly. It has been adopted for performance-critical and security-sensitive components inside large projects: parts of Firefox, pieces of the Linux kernel, and infrastructure at companies like Cloudflare, Amazon, and Microsoft. Its combination of speed and safety makes it attractive anywhere a bug could be catastrophic or a garbage collector would be too costly.

## Strengths and criticisms

Rust has repeatedly been voted the most loved programming language in the annual Stack Overflow developer survey. Developers praise its safety guarantees, its tooling, and its helpful compiler error messages. The most common criticism is the steep learning curve: the borrow checker rejects many programs that would compile in other languages, and learning to satisfy it takes time. Compile times are also slower than some competing languages. The community generally treats these as worthwhile trade-offs for the guarantees Rust provides.
