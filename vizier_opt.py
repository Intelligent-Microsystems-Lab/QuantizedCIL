from vizier.service import clients
from vizier.service import pyvizier as vz

import uuid
import subprocess
import time
import pandas as pd


# Algorithm, search space, and metrics.
study_config = vz.StudyConfig(algorithm='EAGLE_STRATEGY')

study_config.search_space.root.add_float_param('init_dyn_scale', 0.5, 5)
study_config.search_space.root.add_float_param('dyn_scale', 0.5, 5)
# study_config.search_space.root.add_float_param('quantile', .5, 1.1)
study_config.search_space.root.add_float_param('lr', .001, 8.)


study_config.metric_information.append(
    vz.MetricInformation('AUC', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(
    study_config, owner='cschaef6', study_id='hyperP')

final_rdict = {}

for i in range(50):
  suggestions = study.suggest(count=10)

  res_dict = {}
  # start jobs

  for suggestion in suggestions:
    params = suggestion.parameters

    res_dict[suggestion._id] = {}
    res_dict[suggestion._id]['uuid'] = str(uuid.uuid4())

    final_rdict[suggestion._id] = suggestion.parameters

    bashCommand = "qsub -o " + ' vizier_quant_cont/' + res_dict[suggestion._id]['uuid'] + '.log' + " -e " + ' vizier_quant_cont/' + res_dict[suggestion._id]['uuid'] + '.err' + " run_mnist.script " + str(
        params['lr']) + ' ' + str(params['init_dyn_scale']) + ' ' + str(params['dyn_scale']) + ' vizier_quant_cont/' + res_dict[suggestion._id]['uuid'] + '.txt'
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
        continue
      else:
        # job done or crashed
        auc = 0
        with open('vizier_quant_cont/' + v['uuid'] + '.log', 'r') as f:
          lines = f.readlines()
          for line in reversed(lines):
            if 'CNN top1 curve: [' in line:
              vals = line.split('[')[-1]
              vals = vals.split(']')[0]
              vals = vals.split(',')
              auc = sum([float(x) for x in vals])
              # import pdb; pdb.set_trace()
              # auc = float(vals[0][:-1]) + float(vals[1][:-2])
              break

        for sug in suggestions:
          if sug._id == k:
            sug.complete(vz.Measurement({'AUC': auc}))
            rm_list.append(k)
            final_rdict[sug._id]['auc'] = auc
            print('Job completed (' + v['uuid'] + '): '
                  + str(sug.parameters) + ' final auc: ' + str(auc))

    for i in rm_list:
      del res_dict[i]


pd.DataFrame.from_dict(final_rdict).transpose().to_csv('final_vizier.csv')
