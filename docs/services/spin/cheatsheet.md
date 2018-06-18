# Spin CLI Cheatsheet

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

## CLI commands

The Rancher CLI must be used from a NERSC system, such as Cori, Edison or
Denovo.  NERSC provides a modified version of the Rancher CLI, and some
subcommands have been removed. For detailed documentation about using the
Rancher CLI at NERSC, see [tips_and_examples/](Spin Tips & Examples).

Rancher official CLI documentation can be found at
https://rancher.com/docs/rancher/v1.6/en/cli/ & a detailed list of CLI commands
can be found at https://rancher.com/docs/rancher/v1.6/en/cli/commands/ .

| Command                                   |  Description  |
| ----------------------                    | ------------- |
| `module load spin`                        | Load module to access Rancher CLI |
| `export RANCHER_ENVIRONMENT=dev-cattle`   | Specify the environment to be used |
| `spin-keygen.sh`                          | Generate your API keys to connect to Spin. Usually only done once. |
| `rancher config`                          | Configure the Rancher CLI. Usually only done once. |
| `rancher help`                            | Show a list of commands |
| `rancher help command`                    | Show help for one command |
| **Operations on stacks**                  |
| `rancher start|stop|restart [stack name]` | Start, Stop or Restart an entire stack |
| `rancher stack ls`                        | List all active & inactive stacks belonging to you |
| `rancher rm [stack name] --type stack`    | Remove an entire stack. **USE WITH CAUTION** |
| `rancher up`                              | Bring all services up. Rancher will name the stack after the current working directory, and looks for `docker-compose.yml` in the current working directory. Logs are sent to the foreground-- type **Control C** to send the logs to the background. |
| `rancher up -d`                           | Send logs to the background |
| `rancher up --file .../docker-compose.yml.dev` | Bring all services up, specifying an alternate name for and path for `docker-compose.yml` |
| `rancher up --name [stack name]`          | Bring all services up, specifying an alternative project name |
| `rancher up --upgrade`                    | Upgrade the stack if a service has changed |
| `rancher up --force-upgrade`              | Force the stack regardless if a service has changed |
| `rancher up --upgrade --confirm-ugrade`   | Confirm that the upgrade was success and delete old containers |
| `rancher up --upgrade --rollback`         | Rollback to the previous version of the containers |
| `rancher up --render`                     | Render the `docker-compose.yml` file & print output if successful. Useful for syntax checking. |
| `rancher export`                          | Exports the `docker-compose.yml` & `rancher-compose.yml` for a stack |
| `rancher export --file file.tar`          | Exports the `docker-compose.yml` & `rancher-compose.yml` for a stack to a tar file |
| **Operations on services & containers**   |
| `rancher start|stop|restart [stack name]/[service name]` | Start, Stop or Restart one service in your stack |
| `rancher ps`                              | List active services in your stacks |
| `rancher ps --all`                        | List all active & inactive services in your stacks |
| `rancher ps --containers`                 | List active containers in your stacks |
| `rancher ps --containers --all`           | List all active & inactive containers in your stacks |
| `rancher logs [stack name]/[service name]` | View logs for a service (all container instances) |
| `rancher logs [stack name]-[service name]-[instance #]` | View logs for a single container instance of service. The *instance #* can be seen using `rancher ps --containers` |
| `rancher logs --since 1h --timestamps --follow [stack name]/[service name]` | View logs & follow the output, similar to `tail --follow`, and print timestamps. |
| `rancher exec -it [stack name]/[service name] /bin/bash` | Obtain a shell on a container. If the service has more than one container instance, Rancher will ask which instance |
| `rancher exec -it [stack name]-[service name]-[instance #] /bin/bash` | Obtain a shell on a specific container instance of a service. |
| `rancher inspect [stack name]/[service name] | jq '.'` | Print the configuration for a service in JSON, and use `jq` to convert the output to human-friendly format |


## Common approaches when all all fails

TODO Stop the service, remove the containers. Don't remove the service or stack.

