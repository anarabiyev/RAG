# SQL

SQL, which stands for Structured Query Language, is a domain-specific language for managing and querying data held in relational databases. Unlike the general-purpose languages it is often used alongside, SQL is declarative: you describe the data you want and let the database engine work out how to retrieve it.

## History

SQL grew out of research at IBM in the early 1970s. Donald Chamberlin and Raymond Boyce designed the language to work with the relational model, a way of organizing data into tables that had been proposed by the mathematician Edgar F. Codd. The relational model and the query language built on it proved enormously influential. SQL became an ANSI standard in 1986 and an ISO standard shortly after, and although every database vendor has added its own extensions and dialect, the core of the language is remarkably consistent across systems decades later.

## The relational model

In the relational model, data is stored in tables made of rows and columns, and relationships between tables are expressed by shared key values rather than by pointers. A query can combine data from several tables using a join, which matches rows based on those keys. This separation of logical structure from physical storage is the model's core idea: you state the relationships you care about, and the database decides how to store and fetch the underlying bytes.

## Declarative querying

SQL's declarative nature is its defining feature. In an imperative language like Python or Rust you write the step-by-step procedure to get a result; in SQL you write what the result should look like, and a component called the query planner or optimizer chooses an efficient execution strategy, often using indexes to avoid scanning entire tables. Core operations are easy to learn: SELECT reads data, INSERT adds rows, UPDATE changes existing rows, and DELETE removes them. More advanced features include aggregations, subqueries, window functions, and transactions that group several changes into an all-or-nothing unit with ACID guarantees.

## Database systems

SQL is implemented by many relational database systems. PostgreSQL and MySQL are the most widely used open-source options; SQLite is a tiny embedded database that ships inside countless applications and phones; and commercial systems like Microsoft SQL Server and Oracle are common in enterprises. Each speaks a slightly different dialect, but a developer who knows standard SQL can move between them with modest effort.

## SQL and RAG

SQL databases are increasingly relevant to retrieval-augmented generation. Several vector databases used in RAG systems are built as extensions on top of traditional SQL databases: pgvector, for example, adds vector columns and similarity search to PostgreSQL, letting a team store embeddings right next to their relational data and query both with familiar SQL. This means the similarity search at the heart of retrieval, which this project implements by hand with NumPy, can in production be handled by the same database that stores the rest of an application's data.
