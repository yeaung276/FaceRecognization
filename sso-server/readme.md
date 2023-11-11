This is SSO server. This has three endpoint
register will accept the encoding and register the user in the database
when call authenticate endpoint with user id and embedding, the sso will perform authentication based on those encoding and return redirect response if the user match
whoever accepting that redirect can stript that code and call claim endpoint to get the authorization token
# Do the migration with
```alembic upgrade head```