# SQL Query Engine with Extended SQL (EMF Processing)

## Overview

This project implements a custom SQL query processing engine designed to simplify complex OLAP-style queries. Traditional SQL requires multiple joins, subqueries, and views to compare dynamic subsets of data, which can be difficult to write and inefficient to execute.

This system extends SQL using grouping variables and processes queries using a Phi operator based algorithm, enabling cleaner query expression and efficient execution.

---

## Problem Statement

OLAP queries often require multiple aggregates across different subsets of data such as time slices or group comparisons. Standard SQL becomes verbose and computationally expensive due to repeated joins and nested queries.

The goal of this project was to:

- Simplify expression of complex group-based queries
- Reduce reliance on joins and subqueries
- Provide a structured execution model
- Improve query readability and processing efficiency

---

## Solution Approach

The system accepts queries in an Extended SQL (EMF) format and converts them into a Phi representation. The engine then processes this representation using a multi-pass algorithm to compute aggregates efficiently.

---

## System Architecture

The query engine follows a layered architecture:

1. Input Layer → Accepts Extended SQL queries
2. Parser → Converts queries into structured Phi components
3. mf_struct Builder → Creates an in-memory structure for grouping variables
4. Processing Engine → Scans data and computes aggregates
5. Output Layer → Displays final results

This modular design enables clear separation of concerns and extensibility.

---

## Key Features

- Extended SQL query format support
- Phi operator based execution model
- Multi-pass database scanning algorithm
- Dynamic grouping variable handling
- Aggregation across multiple subsets
- Modular architecture design
- Command line execution

---

## Tech Stack

- Python
- PostgreSQL
- psycopg2
- tabulate
- dotenv

---

## Example Query Capability

The engine supports queries that compute aggregates across different group variables and conditions, enabling analysis across multiple states or time slices without complex SQL joins.

---

## Engineering Highlights

- Implemented custom query parsing logic
- Designed multi-pass aggregation algorithm
- Built in-memory grouping structure (mf_struct)
- Applied modular system design principles
- Explored tradeoffs between flexibility and performance
- Designed execution pipeline similar to database internals

---
## How to Run

### Prerequisites

- Python 3.x
- PostgreSQL (if running against database)
- Required libraries:
  - psycopg2
  - python-dotenv
  - tabulate

Install dependencies if needed:

bash
pip install -r requirements.txt

---

## Future Improvements

- Cost-based query optimization
- Dynamic indexing (B+ tree)
- Query validation engine
- Parallel execution
- Web interface for visualization
- Support for nested queries

---

## Learning Outcomes

This project provided hands-on experience with database internals, query execution models, and backend system design. It strengthened understanding of how query planners and execution engines work in real database systems.
