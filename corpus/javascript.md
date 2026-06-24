# JavaScript

JavaScript is a high-level, dynamically typed programming language that is the native language of the web. It is the only language that runs directly in every web browser, which made it indispensable for front-end development, and through Node.js it has become a major server-side language as well.

## History

JavaScript was created by Brendan Eich at Netscape in 1995, reportedly in about ten days. It was first called Mocha, then LiveScript, and finally JavaScript — a marketing decision that has confused people ever since, because the language has little to do with Java. To prevent the language from fragmenting across browsers, it was standardized as ECMAScript, with the specification maintained by Ecma International. A landmark version, ECMAScript 2015 (often called ES6), modernized the language with classes, modules, arrow functions, promises, and more, and the standard has been updated on a yearly cadence since.

## Language model

JavaScript is dynamically typed and multi-paradigm, supporting object-oriented, functional, and imperative styles. Its object model is prototype-based rather than class-based, although ES6 added class syntax as a more familiar layer on top. Functions are first-class values and closures are central to how the language is used. JavaScript is famous for a few rough edges in its type system, such as implicit type coercion and the difference between the `==` and `===` equality operators, quirks that survive for backward compatibility but are well understood by experienced developers.

## The event loop

JavaScript is single-threaded and event-driven. Rather than blocking while waiting for a network request or a timer, it registers callbacks and continues; a mechanism called the event loop runs those callbacks when the awaited work completes. Asynchronous code was originally written with nested callbacks, which became hard to read, then with promises, and now most commonly with async/await syntax that makes asynchronous code look sequential. This concurrency model is well suited to I/O-heavy workloads like web servers handling many simultaneous connections.

## Runtimes and ecosystem

For its first fourteen years JavaScript ran only in browsers. That changed in 2009 when Ryan Dahl released Node.js, which paired Google's fast V8 engine with libraries for file and network access, letting the same language run on the server. Node's npm registry is now one of the largest package ecosystems of any language. TypeScript, a superset that adds optional static typing and compiles down to JavaScript, has become extremely popular for larger codebases. Newer runtimes like Deno and Bun aim to modernize the server-side experience.

## Use cases

JavaScript powers interactive front-ends through frameworks such as React, Angular, and Vue, and back-ends through Node and frameworks like Express. The same language also reaches mobile apps via React Native and desktop apps via Electron, so a single team can use one language across the whole stack. This ubiquity, combined with the fact that it cannot be avoided on the web, keeps JavaScript among the most widely used languages in the world.
