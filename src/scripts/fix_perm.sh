#!/bin/bash
# fix_perm.sh

for dir in /home/cs-25-344/*; do
    echo "Fixing $dir"
    
    # Fix group ownership and core permissions
    chgrp -R "egr cs-25-344" "$dir"
    chmod -R g+rwX,o-rwx "$dir"
    
    # Make all .sh scripts executable by both user and group
    find "$dir" -type f -name "*.sh" -exec chmod ug+x {} \;
done

echo "Done."
