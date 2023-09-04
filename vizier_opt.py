from vizier.service import clients
from vizier.service import pyvizier as vz

import uuid
import subprocess
import time

# Objective function to maximize.
def evaluate(params) -> float:
  return w**2 - y**2 + x * ord(z)

# Algorithm, search space, and metrics.
study_config = vz.StudyConfig(algorithm='EAGLE_STRATEGY')
# study_config.search_space.root.add_float_param('w', 0.0, 5.0)
# study_config.search_space.root.add_int_param('x', -2, 2)
# study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])


study_config.search_space.root.add_float_param('tolerance_update', 0.0, 1.01)
study_config.search_space.root.add_float_param('calib_maxf_err', 0.0, 128.0)
study_config.search_space.root.add_float_param('lr', 1e-9, .1, scale_type=vz.ScaleType.LOG)
study_config.search_space.root.add_categorical_param('quant_scheme', ['e3m0', 'e2m1', 'four', 'eight', 'sixt'])



study_config.metric_information.append(vz.MetricInformation('AUC', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')



for i in range(30):
  suggestions = study.suggest(count=4)

  res_dict = {}
  # start jobs

  
  for suggestion in suggestions:
    params = suggestion.parameters

    res_dict[suggestion._id] = {}
    res_dict[suggestion._id]['uuid'] = str(uuid.uuid4())

    # --err_bits=$1 --l_rate_fn=$2 --qerr_update_tol=$3 --err_calib_method=$4 --save_dir=$5 
    bashCommand = "qsub -o " +  ' ../vizier_quant_cont/' +  res_dict[suggestion._id]['uuid'] + '.log' + " -e " +  ' ../vizier_quant_cont/' +  res_dict[suggestion._id]['uuid'] + '.err' + " crc.script " + params['quant_scheme'] + ' ' + str(params['lr']) + ' ' + str(params['tolerance_update']) + ' maxf_' + str(params['calib_maxf_err']) + ' ../vizier_quant_cont/' +  res_dict[suggestion._id]['uuid'] + '.txt'
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if str(output).split(' ')[2].isdigit():
      res_dict[suggestion._id]['crc_id'] = str(output).split(' ')[2]
    else:
      del res_dict[suggestion._id]
      suggestion.complete(vz.Measurement({'AUC': 0}))

    print('Job launched successfully.')


  while res_dict != {}:
    # wait 10s
    time.sleep(10)

    rm_list = []
    for k, v in res_dict.items():
      bashCommand = "qstat -u cschaef6"
      process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
      output, error = process.communicate()

      if v['crc_id'] in str(output):
        pass
      else:
        # job done or crashed
        try:
          with open('../vizier_quant_cont/' +  v['uuid'] +  '.txt', 'r') as the_file:
            auc = float(the_file.read())
        except:
          print('Trial failed... no final file.')  
          auc = 0

        for sug in suggestions:
          if sug._id == k:
            sug.complete(vz.Measurement({'AUC': auc}))
            rm_list.append(k)
            print('Job completed (' + v['uuid'] + '): ' + str(sug.parameters) + ' final auc: ' + str(auc))

    for i in rm_list:
      del res_dict[i]


  #   objective = evaluate(params['w'], params['x'], params['y'], params['z'])

  # # wait for results

  # # get results
  # for suggestion in suggestions:
  #   suggestion.complete(vz.Measurement({'AUC': res_dict[suggestion._id]}))
  