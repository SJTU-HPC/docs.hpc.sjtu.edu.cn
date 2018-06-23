# Spin Cheatsheet

## Naming conventions

* An ***Application Stack*** contains one or more ***services***, each
  performing a distinct function. Services are named as **[Stack
  Name]/[Service Name]**, such as **sciencestack/web** or **sciencestack/db**.
* Services contain one or more instances of itself, called ***containers***.
  Containers are named as **[Stack Name]-[Service Name]-[Instance #]** where
  ***Instance #*** is the number of that container
  instance, such as **sciencestack-web-1** and **sciencestack-web-2**.

## List of ports and ranges

TODO

## Rancher CLI Cheatsheet

The Rancher CLI must be used from a NERSC system, such as Cori, Edison or
Denovo.  NERSC provides a modified version of the Rancher CLI which is
optimized for the NERSC environment, and some subcommands have been removed.
For detailed documentation about using the Rancher CLI at NERSC, see
[Spin Tips & Examples](tips_and_examples).

Rancher official CLI documentation can be found at
https://rancher.com/docs/rancher/v1.6/en/cli/ & a detailed list of CLI commands
can be found at https://rancher.com/docs/rancher/v1.6/en/cli/commands/ .

### Global commands

| Command                                     | Description                     |
| -----------------------------------------   | ------------------------------- |
| `module load spin`                          | Load the module to access the Spin CLI tools, such as `rancher` |
| `export RANCHER_ENVIRONMENT=dev-cattle`     | Specify the Rancher environment to be used |
| `rancher environment`                       | Print all environments that your account has access too |
| `spin-keygen.sh`                            | Generate your API keys to connect to Spin. Usually only done once. |
| `rancher config`                            | Configure the Rancher CLI. Usually only done once. |
| `rancher help`                              | Show a list of commands |
| `rancher help [command]`                    | Show help for one command |
| `rancher [command] --help`                  | Show help for one command |
| `rancher [command] --config .../cli.json`   | Specify an alternate CLI client configuration file before running `command`. |

### Operations on stacks

| Command                                   | Description                     |
| ----------------------------------------- | ------------------------------- |
| `rancher start|stop|restart [stack name]` | Start, Stop or Restart an entire stack |
| `rancher stack ls`                        | List all active & inactive stacks belonging to you |
| `rancher rm [stack name] --type stack`    | Remove an entire stack. **USE WITH CAUTION** |
| **Creating and upgrading stacks** <br> - See footnote [^1] for `rancher up`'s default behavior |
| `rancher up`                              | Create & start the stack. Requires `docker-compose.yml`. |
| `rancher up -d`                           | Create & start the stack. Send logs to the background. |
| `rancher up --file .../docker-compose.yml.dev` | Create & start the stack, specifying an alternate Docker Compose file |
| `rancher up --name [stack name]`          | Create & start the stack, specifying an alternative project name |
| `rancher up --upgrade`                    | Upgrade the stack if a service in `docker-compose.yml` has changed |
| `rancher up --force-upgrade`              | Force the stack regardless if a service has changed |
| `rancher up --upgrade --confirm-ugrade`   | Confirm that the upgrade was success and delete old containers |
| `rancher up --upgrade --rollback`         | Rollback to the previous version of the containers |
| `rancher up --render`                     | Read your `docker-compose.yml` file & print output if successful.<br>Useful for syntax checking and for variable interpolation. |
| `rancher export [stack name]`                 | Export the stack's Docker & Rancher Compose files to a subdirectory named `[stack name]` |
| `rancher export [stack name] --file file.tar` | Export the stack's Docker & Rancher Compose files to a tar file |
| **Rancher Secrets**                      |
| `rancher secret ls`                       | List all secrets owned by you |
| `rancher secret create [secret name] file-with-secret`    | Create a secret named [secret name] using the value read from the file `file-with-secret` |
| `echo MyPassword | rancher secret create [secret name] -` | Create a secret named [secret name], read from standard input |
| `rancher secret create [secret name] - <<< MyPassword`    | Create a secret named [secret name], read from standard input |
| `rancher secret rm [secret name]`         | Remove the secret `[secret name]` if owned by you |
| **Rancher Volumes**                       |
| `rancher volume ls`                       | List all volumes owned by you |
| `rancher volume create --driver rancher-nfs [service name].[stack name]` | Create a volume on the Rancher NFS server named [service name].[stack name] |
| `rancher volume rm [service name].[stack name]` | Remove a volume owned by you |

### Operations on services & containers

| Command                                   | Description                     |
| ----------------------------------------- | ------------------------------- |
| `rancher start|stop|restart [stack name]/[service name]` | Start, Stop or Restart one service in your stack |
| `rancher ps`                              | List active services in all of your stacks |
| `rancher ps --all`                        | List all active & inactive services in all of your stacks |
| `rancher ps --containers`                 | List active containers in your stacks |
| `rancher ps --containers --all`           | List all active & inactive containers in your stacks |
| `rancher ps | grep [stack name]`          | List active services in one stack, using `grep` to show just one stack |
| `rancher ps --format '{{.Service.Id}} {{.Service.Name}}/{{.Service.Name}}'` | Format the output of `rancher ps` to only print the ID and name of the services |
| `rancher logs [stack name]/[service name]` | View logs for a service, including all container instances of that service. |
| `rancher logs [stack name]-[service name]-[instance #]` | View logs for a single container instance of service.<br>The *instance #* can be seen using `rancher ps --containers` |
| `rancher logs --since 1h --timestamps --follow [stack name]/[service name]` | View logs for a service & follow the output,<br>similar to `tail --follow`, and print timestamps. |
| `rancher exec -it [stack name]/[service name] /bin/bash`              | Obtain a shell on a container.<br>If the service has more than one container instance, Rancher will ask which instance. |
| `rancher exec -it [stack name]-[service name]-[instance #] /bin/bash` | Obtain a shell on a specific container instance of a service. |
| `rancher inspect [stack name]/[service name] | jq`                    | Print the configuration for a service in JSON, and use `jq` to convert the output to human-friendly format |
| `rancher inspect [stack name]/[service name] | jq '.' | grep 'value'` | Print the configuration for a service in JSON, use `jq` to apply the filter `'.'`,<br>and search for a value using `grep`. The filter is required when passing to standard out. |
| `rancher scale [stack name]/[service name]=2` | Set number of containers to run for a service. In this case, the service is scaled to two containers. |

## Common approaches when all else fails

TODO Stop the service, remove the containers. Don't remove the service or stack.

footnotes:
[^1]: `rancher up` has several default behaviors. See the chart above to override these behaviors.

    * `docker-compose.yml` is required. `rancher-compose.yml` is optional, and rarely used.
    * `rancher up` reads the stack configuration from `docker-compose.yml` in
      the current working directory.
    * Rancher will name the stack after the current working directory.
    * Application logs are sent to the foreground. When logs are in the
      foreground, type **Control C** to send the logs to the background. The
      stack will continue running in the background.
