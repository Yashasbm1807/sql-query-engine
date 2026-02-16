1. Run generator file with 'python emf.py'
2. Follow prompts to either read input from file or enter inputs via prompts
3. Run generated code with 'run_emf_query.py'
4. See output on terminal or see output written in table_output.txt

Note1 : There are no additional requirements apart from the very basic, psycopg2, python-dotenv and tabulate. 
These are mentioned in the requirements which can be installed using pip install -r requirements.txt.
But it is more than likely your system already has these, in which case it can be skipped and code can be run directly.

Note2 : The only relevant python file is the generator named emf.py. The other files are examples in the following format.
The emf input to be read is 'query{x}emf.txt' for number x.
The sql query to generate output is 'query{x}_sql.txt' for number x.
The generated output from sql is saved in JPEG format named 'query{x}sql.jpg' for number x.