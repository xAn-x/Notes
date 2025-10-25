### IAM (Identity & Access Management): Users & Groups

**Global Service** for managing user and groups(multiple users, not other groups) in an org. for assigning permisions or access to resource (what one can access or do)

These policies define permissions of users

In AWS u apply the **least privilege principle**: don't give more permissions than a user needs.

Best Practice: Use IAM role to access AWS services.

**IAM Policies Stucture**

  

IAM policies are JSON files defines what permissions does a user/account/policy.

  

```json

{

	"Version": "2012-10-17", // policy-lang-version
	
	"Id": "S3-Dev-Perm", // identifier for policy(optional)

	"Statement": [ // one or more statements (required)
	
		{
			"Sid": "1", // identifier for statement (optional)
			
			"Effect":"Allow", // weather to 'allow/deny' the access
			
			"Principal": {
			
			"AWS": ["account/user/role"]
		},
		"Action":[ // list of actions this policy allows or denies
			"s3:GetObject",
			
			"s3:PutObject"
		],
		
		"Resource":[
			"bucket/*"
		]
	]
}

```

**AWS CLI**

Give you access to public API of aws on your cmd, so that you can manage resources directly from your local device.

To access these services , first you need the access key that authenticate that yes you have the permissions.

⦁ __To create Access keys__

   > IAM > User > {select user} > create access key > CLI > {download csv/ have creds in hand}


⦁ Configure AWS CLI

```cmd

>> aws configure {then put key and password,then u can run cmds}

```


**IAM Roles for Services**

⦁ Some AWS service will need to perform actions on your behalf & to do so we will assign permissions to AWS services with IAM Roles.

>[!Note]
>they will not be used by users but the aws services (eg. we add roles to AWS glue which tells AWS that using glue we can perform these list of actions


**IAM Security Tools**


⦁ IAM Credentials Report (account-level): report that lists all ur acnt users & status of their various creds.

⦁ IAM Access Advisor (user-level): shows the service perms granted to a user & when those services were last accessed. U can use this info to revise your policies.

#### Shared Responsibility Model for IAM


AWS : Infra (global network security), Config and vulnerability analysis, Compliance validation.

YOU : Users, Groups, Roles, Policies management & monitoring. Enable MFA on all accounts, rotate ur keys etc.