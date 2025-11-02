# .bashrc
# version 1.1

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
alias cp='cp -i'

# Miniconda bin directory location
DIR="$HOME/miniconda3/bin"
DIR2="/data/$USER/miniconda3/bin"

if [ -d "$DIR" ] || [ -d "$DIR2" ]; then

	unset CONDA_AUTO_ACTIVATE_BASE

	if [ -d "$DIR" ]; then

		echo "Using Miniconda3 installed in your Home directory"

		# >>> conda initialize >>>
		# !! Contents within this block are managed by 'conda init' !!
		__conda_setup="$( /home/${USER}/miniconda3/bin/conda 'shell.bash' 'hook' 2> /dev/null)"

		if [ $? -eq 0 ]; then
    			eval "$__conda_setup"
		else
    			if [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        			. "/home/$USER/miniconda3/etc/profile.d/conda.sh"
    			else
        			export PATH="/home/$USER/miniconda3/bin:$PATH"
    			fi
		fi
		unset __conda_setup
		# <<< conda initialize <<<

	else

		echo "Using Miniconda3 installed in your Data directory"

		# >>> conda initialize >>>
                # !! Contents within this block are managed by 'conda init' !!
                __conda_setup="$( /data/${USER}/miniconda3/bin/conda 'shell.bash' 'hook' 2> /dev/null)"

                if [ $? -eq 0 ]; then
                        eval "$__conda_setup"
                else
                        if [ -f "/data/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
                                . "/data/$USER/miniconda3/etc/profile.d/conda.sh"
                        else
                                export PATH="/data/$USER/miniconda3/bin:$PATH"
                        fi
                fi
                unset __conda_setup
                # <<< conda initialize <<<


	fi

else

	echo ""
        # Miniconda is not installed in home or data directory
        echo "Miniconda3 is NOT installed in your Home or Data directory. Please read the portal documentation for details."
        echo ""

fi

