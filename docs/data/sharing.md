# Data sharing

## Unix file permissions

[Unix permissions](https://en.wikipedia.org/wiki/File_system_permissions)

## give/take

NERSC provides two commands: `give` and `take` which are useful for
sharing data between users.

To send a file or path to `<receiving_username>`:
```
nersc$ give -u <receiving_username> <file or directory>
```

To receive a file sent by `<sending_username>`:
```
nersc$ take -u <sending_username> <filename>
```

To take all files from `<sending_username>`:
```
nersc$ take -a -u <sending_username>
```

To see what files `<sending_username>` has sent to you:
```
nersc$ take -u <sending_username>
```

For a full list of options pass the `--help` flag.

## project directories

The [project](/filesystems/project.md) filesystem allows sharing of
data within a project.

## Science Gateways

* [Science gateways](/services/science-gateways.md)

